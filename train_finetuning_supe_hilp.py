#! /usr/bin/env python
import gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from absl import app, flags, logging
from flax.training import checkpoints
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from ml_collections import config_flags
import wandb

from get_hilp_agent import (get_default_config, get_restore_path,
                            load_hilp_agent)
from supe.agents import RM, RND, SACLearner  # NOQA
from supe.data import ChunkDataset, D4RLDataset, ReplayBuffer
from supe.evaluation import evaluate
from supe.pretraining.opal import OPAL
from supe.utils import (add_prefix, check_overlap, combine,
                        view_data_distribution)
from supe.visualization import (get_canvas_image, get_env_and_dataset,
                                plot_data_directions, plot_q_values,
                                plot_rnd_reward, plot_trajectories)
from supe.wrappers import (MaskKitchenGoal, MetaPolicyActionWrapper,
                           TanhConverter, wrap_gym)

logging.set_verbosity(logging.FATAL)

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "hilp", "wandb project name.")
flags.DEFINE_string("env_name", "antmaze-large-diverse-v2", "D4rl dataset name.")
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
flags.DEFINE_string("offline_relabel_type", "gt", "one of [gt/pred/min]")
flags.DEFINE_boolean("use_rnd_offline", False, "Whether to use rnd offline.")
flags.DEFINE_boolean("use_rnd_online", False, "Whether to use rnd online.")
flags.DEFINE_integer(
    "hpolicy_horizon",
    4,
    "each high level action is kept fixed for how many time steps",
)
flags.DEFINE_bool(
    "interpolate", False, "wheter to interolate skills from intermediate states"
)
flags.DEFINE_integer("updates_per_step", 4, "Number of updates per step")
flags.DEFINE_bool("debug", False, "Whether to be in debug mode")
flags.DEFINE_bool("vision", False, "Whether to use vision based environment")

config_flags.DEFINE_config_file(
    "config",
    "configs/rlpd_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "rm_config",
    "configs/rm_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "rnd_config",
    "configs/rnd_config.py",
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
    assert FLAGS.offline_ratio <= 1.0

    wandb.init(project=FLAGS.project_name)
    wandb.config.update(FLAGS)

    if FLAGS.debug:
        FLAGS.max_steps = 1000
        FLAGS.eval_episodes = 1
        FLAGS.start_training = 10
        FLAGS.eval_interval = 10
        FLAGS.log_interval = 10
        FLAGS.save_video = False

    ########### LOWER LEVEL ENVIRONMENT ###########

    env = gym.make(FLAGS.env_name)
    eval_env = gym.make(FLAGS.env_name)

    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env = wrap_gym(env, rescale_actions=True)

    eval_env = wrap_gym(eval_env, rescale_actions=True)

    if "kitchen" in FLAGS.env_name:
        env = MaskKitchenGoal(env)
        env.env.env.env.env.env.env.env.env.REMOVE_TASKS_WHEN_COMPLETE = False
        eval_env = MaskKitchenGoal(eval_env)

    ########### LOWER LEVEL AGENT ###########

    observation_space, action_space = eval_env.observation_space, eval_env.action_space
    observations, actions = observation_space.sample(), action_space.sample()

    rng = jax.random.PRNGKey(FLAGS.seed)

    agent_rng, rng = jax.random.split(rng)
    agent = OPAL.create(
        FLAGS.opal_config,
        agent_rng,
        observations,
        actions,
        chunk_size=FLAGS.hpolicy_horizon,
    )

    skill_agent_config = get_default_config()
    skill_agent_config.env_name = FLAGS.env_name
    hilp_agent_path = get_restore_path(FLAGS.env_name, seed=FLAGS.seed)
    print(hilp_agent_path)

    agent = load_hilp_agent(
        skill_agent_config,
        hilp_agent_path,
        restore_epoch=1000000 if "antmaze" in FLAGS.env_name else 500000,
    )
    skill_dim = skill_agent_config.skill_dim

    ########### META ENVIRONMENT ###########

    rng, episode_rng = jax.random.split(rng)
    meta_env = MetaPolicyActionWrapper(
        env,
        agent,
        episode_rng,
        FLAGS.hpolicy_horizon,
        subtract_one="antmaze"
        in FLAGS.env_name,  # if this is not antmaze, we do not want to subtract 1 from reward
        hilp=True,
        skill_dim=skill_dim,
    )
    meta_env.seed(FLAGS.seed)

    rng, eval_rng = jax.random.split(rng)
    eval_meta_env = MetaPolicyActionWrapper(
        eval_env,
        agent,
        eval_rng,
        FLAGS.hpolicy_horizon,
        eval=True,
        hilp=True,
        skill_dim=skill_dim,
    )
    eval_meta_env.seed(FLAGS.seed + 42)

    original_ds = D4RLDataset(
        env,
        subtract_one="antmaze" in FLAGS.env_name,
        remove_kitchen_goal="kitchen" in FLAGS.env_name,
    )

    tanh_converter = TanhConverter()

    ds = ChunkDataset.create(
        original_ds,
        chunk_size=FLAGS.hpolicy_horizon,
        agent=agent,
        tanh_converter=tanh_converter,
        discount=0.99,
        batch_size=32768,
        label_skills=True,
        debug=FLAGS.debug,
        skill_dim=skill_dim,
        hilp=True,
    )

    if "antmaze" in FLAGS.env_name:

        viz_env, viz_dataset = get_env_and_dataset(FLAGS.env_name)
        coords, S = viz_env.get_coord_list()

    ds_minr = ds.sample(None)["rewards"].min()
    print(f"Dataset minimum reward = {ds_minr}")

    ########### MODELS ###########

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    meta_agent = globals()[model_cls].create(
        FLAGS.seed, observation_space, meta_env.action_space, **kwargs
    )

    meta_replay_buffer = ReplayBuffer(
        meta_env.observation_space,
        meta_env.action_space,
        FLAGS.max_steps,
    )
    meta_replay_buffer.seed(FLAGS.seed)

    if FLAGS.use_rnd_offline or FLAGS.use_rnd_online:
        kwargs = dict(FLAGS.rnd_config)
        model_cls = kwargs.pop("model_cls")
        rnd = globals()[model_cls].create(
            FLAGS.seed + 123,
            meta_env.observation_space,
            meta_env.action_space,
            **kwargs,
        )
    else:
        rnd = None

    if FLAGS.offline_relabel_type == "gt":
        rm = None
    else:
        kwargs = dict(FLAGS.rm_config)
        model_cls = kwargs.pop("model_cls")
        rm = globals()[model_cls].create(
            FLAGS.seed + 123,
            meta_env.observation_space,
            meta_env.action_space,
            **kwargs,
        )

    # Train meta policy
    observation, done = meta_env.reset(), False
    online_trajs = []
    online_traj = [observation]
    env_step = 0
    record_step = 0
    for i in tqdm.tqdm(
        range(0, FLAGS.max_steps + 1, FLAGS.hpolicy_horizon),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
    ):
        record_step += 1
        if i < FLAGS.start_training:
            action = np.clip(
                np.random.rand(skill_dim) * 2 - 1.0, -1.0 + 1e-6, 1.0 - 1e-6
            )
        else:
            action, meta_agent = meta_agent.sample_actions(observation)

        arctanh_action = tanh_converter.from_tanh(action)

        a_action = arctanh_action / np.linalg.norm(arctanh_action)
        next_observation, reward, done, info = meta_env.step(a_action)

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

        if i % 50 == 0 and FLAGS.interpolate:
            results = meta_env.process_buffer()
            for (
                observation_i,
                next_observation_i,
                reward_i,
                done_i,
                mask_i,
                _,  # info_i, unused
                action_i,
            ) in results:
                action_i = tanh_converter.to_tanh(action_i)

                meta_replay_buffer.insert(
                    dict(
                        observations=observation_i,
                        actions=action_i,
                        rewards=reward_i,
                        masks=mask_i,
                        dones=done_i,
                        next_observations=next_observation_i,
                    )
                )

                if rnd is not None and i >= 2 * FLAGS.start_training:
                    rnd, _ = rnd.update(
                        {
                            "observations": observation_i[None],
                            "actions": action_i[None],
                            "next_observations": next_observation_i[None],
                            "rewards": np.array(reward_i)[None],
                            "masks": np.array(mask_i)[None],
                            "dones": np.array(done_i)[None],
                        }
                    )

        if i >= FLAGS.start_training:
            for _ in range(FLAGS.updates_per_step):
                online_batch_size = int(
                    FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio)
                )
                online_batch = meta_replay_buffer.sample(online_batch_size)
                online_batch = online_batch.unfreeze()

                if FLAGS.use_rnd_online:
                    online_rnd_reward = rnd.get_reward(
                        online_batch["observations"], online_batch["actions"]
                    )
                    online_batch["rewards"] += online_rnd_reward

                batch = online_batch

                # append offline batch
                if FLAGS.offline_ratio > 0:
                    offline_batch_size = int(
                        FLAGS.batch_size * FLAGS.utd_ratio * FLAGS.offline_ratio
                    )
                    offline_batch = ds.sample(offline_batch_size)
                    offline_batch = offline_batch.unfreeze()

                    if FLAGS.offline_relabel_type == "gt":
                        pass
                    elif FLAGS.offline_relabel_type == "pred":
                        offline_batch["rewards"] = rm.get_reward(
                            offline_batch["observations"], offline_batch["actions"]
                        )
                        offline_batch["masks"] = rm.get_mask(
                            offline_batch["observations"], offline_batch["actions"]
                        )
                    elif FLAGS.offline_relabel_type == "min":
                        offline_batch["rewards"] = ds_minr * np.ones_like(
                            offline_batch["rewards"]
                        )
                        offline_batch["masks"] = rm.get_mask(
                            offline_batch["observations"], offline_batch["actions"]
                        )
                    else:
                        raise NotImplementedError

                    if FLAGS.use_rnd_offline:
                        offline_rnd_reward, offline_rnd_stats = rnd.get_reward(
                            offline_batch["observations"],
                            offline_batch["actions"],
                            stats=True,
                        )
                        offline_batch["rewards"] += offline_rnd_reward

                    batch = combine(offline_batch, batch)
                meta_agent, update_info = meta_agent.update(batch, FLAGS.utd_ratio)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log(add_prefix("agent/", {k: v}), step=record_step)

        # For consistency with old antmaze experiments, don't have compute to rerun those experiments
        start_training_rm = (
            2 * FLAGS.start_training
            if "antmaze" in FLAGS.env_name
            else FLAGS.start_training
        )
        if i >= start_training_rm and rm is not None:
            # need to remove optimism bias from rewards for training RM
            if rnd is not None:
                if FLAGS.use_rnd_online:
                    online_batch["rewards"] -= online_rnd_reward
                if FLAGS.use_rnd_offline:
                    offline_batch["rewards"] -= offline_rnd_reward
            rm, rm_update_info = rm.update(online_batch, FLAGS.utd_ratio)
            rm_update_info.update(rm.evaluate(offline_batch))

        if i >= 2 * FLAGS.start_training and (rm is not None or rnd is not None):
            if rnd is not None:
                rnd, rnd_update_info = rnd.update(
                    {
                        "observations": observation[None],
                        "actions": action[None],
                        "next_observations": next_observation[None],
                        "rewards": np.array(reward)[None],
                        "masks": np.array(mask)[None],
                        "dones": np.array(done)[None],
                    }
                )

                if FLAGS.use_rnd_offline:
                    rnd_update_info.update(offline_rnd_stats)

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
            offline_batch = ds.sample(FLAGS.batch_size)
            if rnd is not None and "antmaze" in FLAGS.env_name:
                rnd_reward_plot = wandb.Image(
                    plot_rnd_reward(viz_env, offline_batch, rnd)
                )
                wandb.log(
                    {f"visualize/rnd_reward_plot": rnd_reward_plot},
                    step=record_step,
                )

            if "antmaze" in FLAGS.env_name:
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
                hilp=True,
            )

            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=record_step)

            if "antmaze" in FLAGS.env_name:

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
                wandb.log(
                    {f"visualize/offline_data_directions": image}, step=record_step
                )


if __name__ == "__main__":
    app.run(main)
