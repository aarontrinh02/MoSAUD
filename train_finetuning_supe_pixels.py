#! /usr/bin/env python
import gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from absl import app, flags, logging
from flax.core import frozen_dict
from flax.core.frozen_dict import FrozenDict
from flax.training import checkpoints
from gym import spaces
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from ml_collections import config_flags
import wandb

from supe.agents.drq import DrQLearner, PixelICVF, PixelRM, PixelRND  # noqa
from supe.data import (ChunkDataset, D4RLDataset, GCSDataset,
                       MemoryEfficientReplayBuffer)
from supe.evaluation import evaluate
from supe.pretraining.opal import OPAL
from supe.utils import (add_prefix, check_overlap,
                        color_maze_and_configure_camera, combine,
                        view_data_distribution)
from supe.visualization import (get_canvas_image, get_env_and_dataset,
                                plot_data_directions, plot_q_values,
                                plot_rnd_reward, plot_trajectories)
from supe.wrappers import MetaPolicyActionWrapper, TanhConverter, wrap_gym

logging.set_verbosity(logging.FATAL)

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "supe-pixels", "wandb project name.")
flags.DEFINE_string("env_name", "antmaze-large-diverse-v2", "d4rl dataset name.")
flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_integer("seed", 1, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 10000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(3e5), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", 5000, "Number of training steps to start training."
)
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_integer("utd_ratio", 20, "Update to data ratio.")
flags.DEFINE_boolean("use_icvf", False, "Whether to use ICVF")
flags.DEFINE_boolean(
    "critic_with_icvf", False, "Whether to use ICVF to initialize critic"
)
flags.DEFINE_integer("icvf_num_steps", 75001, "Number of steps to train ICVF")
flags.DEFINE_string("offline_relabel_type", "min", "one of [gt/pred/min]")
flags.DEFINE_boolean("use_rnd_offline", False, "Whether to use rnd offline.")
flags.DEFINE_boolean("use_rnd_online", False, "Whether to use rnd online.")
flags.DEFINE_integer(
    "hpolicy_horizon",
    4,
    "each high level action is kept fixed for how many time steps",
)
flags.DEFINE_integer("updates_per_step", 8, "Number of updates per step")
flags.DEFINE_bool("debug", False, "Whether to be in debug mode")
flags.DEFINE_string(
    "load_dir",
    "./opal_checkpoints",
    "Directory to load checkpoints from",
)
flags.DEFINE_boolean(
    "interpolate", False, "Whether to interpolate between high level actions"
)

config_flags.DEFINE_config_file(
    "config",
    "configs/drq_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "rm_config",
    "configs/pixel_rm_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "rnd_config",
    "configs/pixel_rnd_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "opal_config",
    "configs/opal_config.py",
    "File path to the opal hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    wandb.init(project=FLAGS.project_name, mode="online")
    wandb.config.update(FLAGS)

    if FLAGS.debug:
        FLAGS.max_steps = 10000
        FLAGS.eval_episodes = 1
        FLAGS.start_training = 10
        FLAGS.eval_interval = 1000
        FLAGS.log_interval = 10
        FLAGS.save_video = False
        FLAGS.icvf_num_steps = 2001

    rng = jax.random.PRNGKey(FLAGS.seed)

    ########### LOWER ENVIRONMENT ###########

    env = gym.make(FLAGS.env_name)
    eval_env = gym.make(FLAGS.env_name)

    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env = wrap_gym(env, rescale_actions=True, render_image=True)

    eval_env = wrap_gym(eval_env, rescale_actions=True, render_image=True)

    # Update colors
    env = color_maze_and_configure_camera(env)
    eval_env = color_maze_and_configure_camera(eval_env)

    viz_env, viz_dataset = get_env_and_dataset(FLAGS.env_name)
    coords, S = viz_env.get_coord_list()

    ########### DATASET ###########

    if FLAGS.env_name.split("-")[-1][0] in ["2", "3", "4"]:
        dataset_env_name = FLAGS.env_name[:-2]
    else:
        dataset_env_name = FLAGS.env_name

    image_dataset = dict(np.load(f"data/antmaze_topview_6_60/{dataset_env_name}.npz"))

    ds = D4RLDataset(env)

    ds.dataset_dict["observations"] = dict(
        position=ds.dataset_dict["observations"][:, :2],
        state=ds.dataset_dict["observations"][:, 2:],
        pixels=image_dataset["images"][..., None],  # need last dimension for stacking
    )

    ds.dataset_dict["next_observations"] = dict(
        position=ds.dataset_dict["next_observations"][:, :2],
        state=ds.dataset_dict["next_observations"][:, 2:],
        pixels=image_dataset["next_images"][..., None],
    )

    ########### LOWER AGENT ###########

    observation_space = gym.spaces.Dict(
        {
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(27,)),
            "pixels": spaces.Box(low=0, high=255, shape=(64, 64, 3, 1), dtype=np.uint8),
            "position": spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
        }
    )
    action_space = eval_env.action_space
    observations, actions = observation_space.sample(), action_space.sample()
    agent_rng, rng = jax.random.split(rng)

    agent = OPAL.create(
        FLAGS.opal_config,
        agent_rng,
        observations,
        actions,
        chunk_size=FLAGS.hpolicy_horizon,
        cnn=True,
    )

    base_name = FLAGS.env_name
    if len(base_name.split("-")) == 5:
        base_name = base_name[:-2]

    agent = checkpoints.restore_checkpoint(
        FLAGS.load_dir
        + "/"
        + str(base_name)
        + "/vision="
        + str(True)
        + "/seed="
        + str(FLAGS.seed),
        target=agent,
        prefix="checkpoint_",
        step=1000000,
    )

    ########### META ENVIRONMENT ###########

    rng, episode_rng = jax.random.split(rng)
    meta_env = MetaPolicyActionWrapper(
        env,
        agent,
        episode_rng,
        FLAGS.hpolicy_horizon,
    )
    meta_env.seed(FLAGS.seed)

    rng, eval_rng = jax.random.split(rng)
    eval_meta_env = MetaPolicyActionWrapper(
        eval_env,
        agent,
        eval_rng,
        FLAGS.hpolicy_horizon,
        eval=True,
    )
    eval_meta_env.seed(FLAGS.seed + 42)

    ########### RELABEL DATASET ###########
    tanh_converter = TanhConverter()

    ds = ChunkDataset.create(
        ds,
        chunk_size=FLAGS.hpolicy_horizon,
        agent=agent,
        tanh_converter=tanh_converter,
        label_skills=True,
        debug=FLAGS.debug,
    )

    ds_minr = -3.940399  # -1 - 0.99 - 0.99**2 - 0.99**3, hardcoded for performance

    ########### MODELS ###########q

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    meta_agent = globals()[model_cls].create(
        FLAGS.seed,
        observation_space,
        meta_env.action_space,
        lower_agent=agent,
        **kwargs,
    )

    meta_replay_buffer = MemoryEfficientReplayBuffer(
        observation_space, meta_env.action_space, FLAGS.max_steps
    )
    meta_replay_buffer.seed(FLAGS.seed)

    if FLAGS.use_rnd_offline or FLAGS.use_rnd_online:
        kwargs = dict(FLAGS.rnd_config)
        model_cls = kwargs.pop("model_cls")
        kwargs["use_icvf"] = FLAGS.use_icvf
        rnd = globals()[model_cls].create(
            FLAGS.seed + 123,
            observation_space,
            meta_env.action_space,
            lower_agent=agent,
            **kwargs,
        )
    else:
        rnd = None

    if FLAGS.offline_relabel_type == "gt":
        rm = None
    else:
        kwargs = dict(FLAGS.rm_config)
        model_cls = kwargs.pop("model_cls")
        kwargs["use_icvf"] = FLAGS.use_icvf
        rm = globals()[model_cls].create(
            FLAGS.seed + 123,
            observation_space,
            meta_env.action_space,
            lower_agent=agent,
            **kwargs,
        )

    # Pre-training
    record_step = 0

    if FLAGS.use_icvf:
        assert (
            rm is not None or rnd is not None
        ), "ICVF is not needed in this configuration"
        observation_space_pixels = gym.spaces.Dict(
            {
                "pixels": spaces.Box(
                    low=0, high=255, shape=(64, 64, 3, 1), dtype=np.uint8
                ),
            }
        )
        icvf = PixelICVF.create(
            FLAGS.seed,
            observation_space_pixels,
            meta_env.action_space,
            pixel_keys=("pixels",),
            **dict(FLAGS.config),
        )
        gc_ds = GCSDataset(ds, **GCSDataset.get_default_config())

        for i in tqdm.trange(
            FLAGS.icvf_num_steps, smoothing=0.1, disable=not FLAGS.tqdm
        ):

            record_step += 1
            batch = gc_ds.sample(FLAGS.batch_size)

            icvf, update_info = icvf.update(frozen_dict.freeze(batch), 1)
            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"icvf-training/{k}": v}, step=record_step)

        replace_keys = ["pixel_encoder"]
        assert (
            "pixel_encoder" in icvf.net.params.keys()
        ), f"pixel_encoder not in {icvf.net.params.keys()}"
        assert (
            "pixel_encoder" in rnd.net.params.keys()
        ), f"pixel_encoder not in {rnd.net.params.keys()}"
        assert (
            "pixel_encoder" in rm.r_net.params.keys()
        ), f"pixel_encoder not in {rm.r_net.params.keys()}"
        replace = {k: icvf.net.params[k] for k in replace_keys}

        if FLAGS.critic_with_icvf:
            # replace pixel_encoder in critic, sufficient since actor and critic share pixel_encoder
            assert (
                "pixel_encoder" in agent.critic.params.keys()
            ), f"pixel_encoder not in {agent.critic.params.keys()}"
            agent = agent.replace(
                critic=agent.critic.replace(params=replace),
            )

        if rnd is not None:
            new_params = FrozenDict(rnd.net.params).copy(add_or_replace=replace)
            new_frozen_params = FrozenDict(rnd.frozen_net.params).copy(
                add_or_replace=replace
            )
            rnd = rnd.replace(
                net=rnd.net.replace(params=new_params),
                frozen_net=rnd.frozen_net.replace(params=new_frozen_params),
            )

        if rm is not None:
            new_params = FrozenDict(rm.r_net.params).copy(add_or_replace=replace)
            rm = rm.replace(r_net=rm.r_net.replace(params=new_params))

    ds_iterator = ds.get_iterator(
        queue_size=2,
        sample_args={
            "sample_shape": int(
                FLAGS.batch_size * FLAGS.utd_ratio * FLAGS.offline_ratio
            ),
        },
    )

    meta_replay_buffer_iterator = meta_replay_buffer.get_iterator(
        queue_size=2,
        sample_args={
            "batch_size": int(
                FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio)
            ),
        },
    )

    # Train meta policy
    observation, done = meta_env.reset(), False
    online_trajs = []
    online_traj = [observation]
    env_step = 0
    for i in tqdm.tqdm(
        range(0, FLAGS.max_steps + 1, FLAGS.hpolicy_horizon),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
    ):
        record_step += 1
        if i < FLAGS.start_training:
            curr_rng, rng = jax.random.split(rng)
            action = agent.prior_model({k: v[None, :] for k, v in observation.items()})[
                0
            ].sample(seed=curr_rng)
            action = tanh_converter.to_tanh(action)
        else:
            action, meta_agent = meta_agent.sample_actions(observation)

        arctanh_action = tanh_converter.from_tanh(action)
        next_observation, reward, done, info = meta_env.step(arctanh_action)

        env_step += FLAGS.hpolicy_horizon

        online_traj.append(next_observation)

        timelimit_stop = "TimeLimit.truncated" in info

        if not done or timelimit_stop:
            mask = 1.0
        else:
            mask = 0.0

        meta_replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )

        if i % 50 == 0 and FLAGS.interpolate:  # process buffer every 50 steps
            results = meta_env.process_buffer()
            for (
                observation_i,
                next_observation_i,
                reward_i,
                done_i,
                mask_i,
                _,  # info_i, unused
                latent_i,
            ) in results:
                latent_i = tanh_converter.to_tanh(latent_i)

                meta_replay_buffer.insert(
                    dict(
                        observations=observation_i,
                        actions=latent_i,
                        rewards=reward_i,
                        masks=mask_i,
                        dones=done_i,
                        next_observations=next_observation_i,
                    )
                )
                if rnd is not None and i >= 2 * FLAGS.start_training:
                    rnd, _ = rnd.update(
                        {
                            "observations": observation_i,
                            "actions": latent_i,
                            "next_observations": next_observation_i,
                            "masks": np.array(mask_i),
                            "dones": np.array(done_i),
                        }
                    )

        if i >= FLAGS.start_training:
            for _ in range(FLAGS.updates_per_step):
                online_batch = next(meta_replay_buffer_iterator)

                if FLAGS.use_rnd_online:
                    online_rnd_rewards = rnd.get_reward(online_batch)
                    online_batch = online_batch.copy(
                        add_or_replace={
                            "rewards": online_batch["rewards"] + online_rnd_rewards
                        }
                    )

                batch = online_batch

                # append offline batch
                if FLAGS.offline_ratio > 0:
                    offline_batch = next(ds_iterator)

                    if FLAGS.offline_relabel_type == "gt":
                        pass
                    elif FLAGS.offline_relabel_type == "pred":
                        rewards = rm.get_reward(offline_batch)
                        masks = rm.get_mask(offline_batch)
                    elif FLAGS.offline_relabel_type == "min":
                        rewards = ds_minr * np.ones_like(offline_batch["rewards"])
                        masks = rm.get_mask(offline_batch)
                    else:
                        raise NotImplementedError

                    if FLAGS.offline_relabel_type != "gt":
                        offline_batch = offline_batch.copy(
                            add_or_replace={"rewards": rewards, "masks": masks}
                        )

                    if FLAGS.use_rnd_offline:
                        offline_rnd_rewards, rnd_stats = rnd.get_reward(
                            offline_batch, stats=True
                        )

                        offline_batch = offline_batch.copy(
                            add_or_replace={
                                "rewards": offline_batch["rewards"]
                                + offline_rnd_rewards
                            }
                        )

                    batch = FrozenDict(combine(offline_batch, batch))

                meta_agent, update_info = meta_agent.update(batch, FLAGS.utd_ratio)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log(add_prefix("agent/", {k: v}), step=record_step)

        if i >= 2 * FLAGS.start_training and (rm is not None or rnd is not None):
            if rnd is not None:  # fix reward labels to not be optimistic anymore
                online_batch = online_batch.copy(
                    add_or_replace={
                        "rewards": online_batch["rewards"] - online_rnd_rewards
                    }
                )
                if FLAGS.use_rnd_offline:
                    offline_batch = offline_batch.copy(
                        add_or_replace={
                            "rewards": offline_batch["rewards"] - offline_rnd_rewards
                        }
                    )

            if rm is not None:
                rm, rm_update_info = rm.update(online_batch, FLAGS.utd_ratio)

            if rnd is not None:
                rnd, rnd_update_info = rnd.update(
                    {
                        "observations": observation,
                        "actions": action,
                        "next_observations": next_observation,
                        "rewards": np.array(reward),
                        "masks": np.array(mask),
                        "dones": np.array(done),
                    }
                )

                if FLAGS.use_rnd_offline:
                    rnd_update_info.update(rnd_stats)

            if i % FLAGS.log_interval == 0:
                if rm is not None:
                    for k, v in rm_update_info.items():
                        wandb.log(add_prefix("rm/", {k: v}), step=record_step)
                if rnd is not None:
                    for k, v in rnd_update_info.items():
                        wandb.log(add_prefix("rnd/", {k: v}), step=record_step)

        if i % FLAGS.log_interval == 0:
            wandb.log({"env_step": env_step}, step=record_step)

        observation = next_observation

        if done or timelimit_stop:
            online_trajs.append({"observation": np.stack(online_traj, axis=0)})
            observation, done = meta_env.reset(), False
            online_traj = [observation]
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log(add_prefix("episode/", {decode[k]: v}), step=record_step)

        if i % FLAGS.eval_interval == 0:
            if (
                i >= FLAGS.start_training
                and rnd is not None
                and FLAGS.offline_ratio > 0
            ):
                if rnd is not None:
                    rnd_reward_plot = wandb.Image(
                        plot_rnd_reward(viz_env, offline_batch, rnd)
                    )
                    wandb.log(
                        {f"visualize/rnd_reward_plot": rnd_reward_plot},
                        step=record_step,
                    )
                q_value_plot = wandb.Image(
                    plot_q_values(viz_env, offline_batch, meta_agent)
                )
                wandb.log({f"visualize/q_value_plot": q_value_plot}, step=record_step)

            eval_info, trajs = evaluate(
                meta_agent,
                eval_meta_env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
                tanh_converter=tanh_converter,
            )

            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=record_step)

            num_overlapped = 0
            for x, y in coords:
                coord = jnp.array([x, y])
                overlapped = False
                for batch in meta_replay_buffer.get_iter(FLAGS.batch_size):
                    if check_overlap(coord, batch["observations"], S / 2):
                        overlapped = True
                        break
                if overlapped:
                    num_overlapped += 1
            wandb.log({"coverage": num_overlapped / len(coords)}, step=record_step)

            fig = plt.figure(tight_layout=True, figsize=(4, 4), dpi=200)
            canvas = FigureCanvas(fig)
            plot_trajectories(viz_env, viz_dataset, online_trajs, fig, plt.gca())
            online_trajs = []
            image = wandb.Image(get_canvas_image(canvas))
            wandb.log({f"visualize/trajs": image}, step=record_step)
            plt.close(fig)

            data_distribution_im = view_data_distribution(viz_env, ds)
            image = wandb.Image(data_distribution_im)
            wandb.log({f"visualize/offline_data_dist": image}, step=record_step)

            data_directions_im = plot_data_directions(viz_env, ds)
            image = wandb.Image(data_directions_im)
            wandb.log({f"visualize/offline_data_directions": image}, step=record_step)


if __name__ == "__main__":
    app.run(main)
