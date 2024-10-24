#! /usr/bin/env python
import os

import gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint
import tqdm
from absl import app, flags, logging
from flax.core import frozen_dict
from flax.core.frozen_dict import FrozenDict
from flax.training import orbax_utils
from gym import spaces
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from ml_collections import config_flags
import wandb

from supe.agents.diffusion import DiffusionBC
from supe.agents.drq import DrQLearner, PixelICVF, PixelRM, PixelRND  # noqa
from supe.data import D4RLDataset, GCSDataset, MemoryEfficientReplayBuffer
from supe.evaluation import evaluate, sample_evaluate
from supe.utils import (add_prefix, check_overlap,
                        color_maze_and_configure_camera, combine,
                        view_data_distribution)
from supe.visualization import (get_canvas_image, get_env_and_dataset,
                                plot_data_directions, plot_q_values,
                                plot_rnd_reward, plot_trajectories)
from supe.wrappers import wrap_gym

logging.set_verbosity(logging.FATAL)

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "explore-pixels", "wandb project name.")
flags.DEFINE_string("env_name", "antmaze-large-diverse-v2", "d4rl dataset name.")
flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_integer("seed", 1, "Random seed.")
flags.DEFINE_integer("eval_episodes", 100, "Number of episodes used for evaluation.")
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
flags.DEFINE_string("offline_relabel_type", "gt", "one of [gt/pred/min]")
flags.DEFINE_boolean("use_rnd_offline", False, "Whether to use rnd offline.")
flags.DEFINE_boolean("use_rnd_online", False, "Whether to use rnd online.")
flags.DEFINE_integer("updates_per_step", 2, "Number of updates per step")
flags.DEFINE_bool("debug", False, "Whether to be in debug mode")
flags.DEFINE_integer("diff_bc_steps", 3000000, "Number of steps to train BC")

flags.DEFINE_float("jsrl_ratio", 0.0, "JSRL ratio")
flags.DEFINE_float("jsrl_discount", 0.99, "Probability of continuing the rollout")

config_flags.DEFINE_config_file(
    "config",
    "configs/rlpd_pixels_config.py",
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
    "diff_config",
    "configs/diff_pixels_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    wandb.init(project=FLAGS.project_name)
    wandb.config.update(FLAGS)

    if FLAGS.debug:
        FLAGS.max_steps = 10000
        FLAGS.eval_episodes = 1
        FLAGS.start_training = 10
        FLAGS.eval_interval = 1000
        FLAGS.log_interval = 10
        FLAGS.save_video = False
        FLAGS.icvf_num_steps = 2001
        FLAGS.diff_bc_steps = 1000

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

    ds = D4RLDataset(env, delete_traj_ends=True)

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

    ds_iterator = ds.get_iterator(
        sample_args={
            "batch_size": int(FLAGS.batch_size * FLAGS.utd_ratio * FLAGS.offline_ratio),
        }
    )

    replay_buffer = MemoryEfficientReplayBuffer(
        observation_space, env.action_space, FLAGS.max_steps
    )
    replay_buffer_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": int(
                FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio)
            ),
        }
    )
    replay_buffer.seed(FLAGS.seed)

    ds_minr = -1  # hardcoded for performance

    ########### MODELS ###########

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed, observation_space, env.action_space, lower_agent=None, **kwargs
    )

    if FLAGS.offline_relabel_type != "gt":
        kwargs = dict(FLAGS.rm_config)
        model_cls = kwargs.pop("model_cls")
        kwargs["use_icvf"] = FLAGS.use_icvf
        rm = globals()[model_cls].create(
            FLAGS.seed + 123,
            observation_space,
            env.action_space,
            lower_agent=None,
            **kwargs,
        )

    else:
        rm = None

    if FLAGS.use_rnd_offline or FLAGS.use_rnd_online:
        kwargs = dict(FLAGS.rnd_config)
        model_cls = kwargs.pop("model_cls")
        kwargs["use_icvf"] = FLAGS.use_icvf
        rnd = globals()[model_cls].create(
            FLAGS.seed + 123,
            observation_space,
            env.action_space,
            lower_agent=None,
            **kwargs,
        )
    else:
        rnd = None

    ########### Diffusion BC Pretraining ###########

    rng = jax.random.PRNGKey(FLAGS.seed)

    example_observations = observation_space.sample()
    example_actions = env.action_space.sample()

    # Pre-training
    record_step = 0
    if FLAGS.jsrl_ratio > 0.0:
        bc_cache_dir = os.path.abspath(
            f"diffusion_checkpoints-vision/dbc-{FLAGS.env_name}/seed-{FLAGS.seed}"
        )
        bc = DiffusionBC.create(
            FLAGS.diff_config,
            jax.random.PRNGKey(FLAGS.seed + 123),
            example_observations,
            example_actions,
        )

        try:
            restored_bc = orbax_checkpointer.restore(bc_cache_dir, item=bc)
        except:
            restored_bc = None

        bc_ds_iterator = ds.get_iterator(
            sample_args={
                "batch_size": int(256),
            }
        )

        if restored_bc is None:
            for _ in tqdm.tqdm(range(FLAGS.diff_bc_steps)):
                record_step += 1
                # batch = ds.sample(16 * 1000)
                batch = next(bc_ds_iterator)
                aux = _ % 10000 == 0
                bc, info = bc.update(batch, utd_ratio=1, aux=aux)
                if aux:
                    for k, v in info.items():
                        wandb.log({f"bc-pretraining/{k}": v}, step=record_step)

                if _ % 1000000 == 0:
                    curr_rng, rng = jax.random.split(rng)
                    eval_info, trajs = sample_evaluate(
                        bc,
                        curr_rng,
                        eval_env,
                        num_episodes=1,
                        save_video=True,
                    )

                    for k, v in eval_info.items():
                        wandb.log({f"bc-evaluation/{k}": v}, step=record_step)

            os.makedirs(os.path.dirname(bc_cache_dir), exist_ok=True)
            orbax_checkpointer.save(bc_cache_dir, bc)
        else:
            bc = restored_bc
            print(f"restored pretrained BC from {bc_cache_dir} successfully")

    # ICVF training and initialize RM and RND with ICVF encoder
    if FLAGS.use_icvf:
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
            env.action_space,
            pixel_keys=("pixels",),
            **dict(FLAGS.config),
        )
        gc_ds = GCSDataset(ds, **GCSDataset.get_default_config())

        for i in tqdm.trange(
            FLAGS.icvf_num_steps, smoothing=0.1, disable=not FLAGS.tqdm
        ):

            record_step += 1
            batch = gc_ds.sample(FLAGS.batch_size)

            icvf, icvf_update_info = icvf.update(frozen_dict.freeze(batch), 1)
            if i % FLAGS.log_interval == 0:
                for k, v in icvf_update_info.items():
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
            new_params = FrozenDict(agent.critic.params).copy(add_or_replace=replace)
            agent = agent.replace(
                critic=agent.critic.replace(params=new_params),
            )
        replace = {k: icvf.net.params[k] for k in replace_keys}

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

    # Training
    observation, done = env.reset(), False
    online_trajs = []
    online_traj = [observation]

    if FLAGS.jsrl_ratio > 0.0:
        curr_rng, rng = jax.random.split(rng)
        rollin_enabled = (
            True if jax.random.uniform(key=curr_rng) < FLAGS.jsrl_ratio else False
        )
    else:
        rollin_enabled = False

    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
    ):
        record_step += 1

        if rollin_enabled:
            curr_rng, rng = jax.random.split(rng)
            action = bc.sample_actions(curr_rng, observation)
            curr_rng, rng = jax.random.split(rng)
            rollin_enabled = (
                True
                if jax.random.uniform(key=curr_rng) <= FLAGS.jsrl_discount
                else False
            )
        else:
            if i < FLAGS.start_training:
                action = env.action_space.sample()
            else:
                action, agent = agent.sample_actions(observation)

        next_observation, reward, done, info = env.step(action)
        reward -= 1
        online_traj.append(next_observation)

        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )
        observation = next_observation

        if done:
            online_trajs.append({"observation": np.stack(online_traj, axis=0)})
            observation, done = env.reset(), False

            if FLAGS.jsrl_ratio > 0.0:
                curr_rng, rng = jax.random.split(rng)
                rollin_enabled = (
                    True
                    if jax.random.uniform(key=curr_rng) < FLAGS.jsrl_ratio
                    else False
                )

            online_traj = [observation]
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log({f"episode/{decode[k]}": v}, step=record_step)

        for _ in range(FLAGS.updates_per_step):
            if i >= FLAGS.start_training:
                online_batch = next(replay_buffer_iterator)

                if FLAGS.use_rnd_online:
                    online_rnd_rewards = rnd.get_reward(
                        frozen_dict.freeze(online_batch)
                    )
                    online_batch = online_batch.copy(
                        add_or_replace={
                            "rewards": online_batch["rewards"] + online_rnd_rewards
                        }
                    )

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

                    batch = FrozenDict(combine(offline_batch, online_batch))
                else:
                    batch = online_batch

                # update the main agent
                agent, update_info = agent.update(batch, FLAGS.utd_ratio)

        if i >= 2 * FLAGS.start_training and (rm is not None or rnd is not None):
            if rnd is not None:  # fix reward labels to not be optimistic anymore
                online_batch = online_batch.copy(
                    add_or_replace={
                        "rewards": online_batch["rewards"] - online_rnd_rewards
                    }
                )
                if FLAGS.offline_ratio > 0:
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

                wandb.log({"env_step": i}, step=record_step)

                for k, v in update_info.items():
                    wandb.log({k: v}, step=record_step)

        if i % FLAGS.eval_interval == 0:
            if i > FLAGS.start_training and rnd is not None and FLAGS.offline_ratio > 0:
                rnd_reward_plot = wandb.Image(
                    plot_rnd_reward(viz_env, offline_batch, rnd)
                )
                wandb.log(
                    {f"visualize/rnd_reward_plot": rnd_reward_plot}, step=record_step
                )
            if FLAGS.offline_ratio > 0:
                q_value_plot = wandb.Image(plot_q_values(viz_env, offline_batch, agent))
                wandb.log({f"visualize/q_value_plot": q_value_plot}, step=record_step)

            eval_info, trajs = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                tanh_converter=None,
                save_video=FLAGS.save_video,
            )

            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=record_step)

            num_overlapped = 0
            for x, y in coords:
                coord = jnp.array([x, y])
                overlapped = False
                for batch in replay_buffer.get_iter(FLAGS.batch_size):
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
