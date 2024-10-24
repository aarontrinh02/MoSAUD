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
from flax.training import orbax_utils
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from ml_collections import config_flags
import wandb

from supe.agents import RM, RND, SACLearner  # NOQA
from supe.agents.diffusion import DiffusionBC
from supe.data import D4RLDataset, ReplayBuffer
from supe.evaluation import evaluate, sample_evaluate
from supe.utils import (add_prefix, check_overlap, combine,
                        view_data_distribution)
from supe.visualization import (get_canvas_image, get_env_and_dataset,
                                plot_data_directions, plot_q_values,
                                plot_rnd_reward, plot_trajectories)
from supe.wrappers import MaskKitchenGoal, wrap_gym


def prefix_metrics(metrics, prefix):
    return {"{}/{}".format(prefix, key): value for key, value in metrics.items()}


logging.set_verbosity(logging.FATAL)

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "explore", "wandb project name.")
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
flags.DEFINE_boolean("debug", False, "Debug mode.")
flags.DEFINE_integer(
    "diff_bc_steps", 3000000, "Number of training steps for diffusion BC."
)

flags.DEFINE_float("jsrl_ratio", 0.0, "probability to rollin diffusion BC online")
flags.DEFINE_float("jsrl_discount", 0.99, "probability to rollin diffusion BC online")

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
    "diff_config",
    "configs/diff_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
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
        FLAGS.diff_bc_steps = 10000
    ########### ENVIRONMENT ###########

    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)

    eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap_gym(eval_env, rescale_actions=True)
    eval_env.seed(FLAGS.seed + 42)

    if "kitchen" in FLAGS.env_name:
        env = MaskKitchenGoal(env)
        env.env.env.env.env.env.env.env.env.REMOVE_TASKS_WHEN_COMPLETE = False
        eval_env = MaskKitchenGoal(eval_env)

    ds = D4RLDataset(
        env,
        subtract_one="antmaze" in FLAGS.env_name,
        remove_kitchen_goal="kitchen" in FLAGS.env_name,
        delete_traj_ends=True,
    )
    ds.seed(FLAGS.seed)

    if "antmaze" in FLAGS.env_name:
        viz_env, viz_dataset = get_env_and_dataset(FLAGS.env_name)
        coords, S = viz_env.get_coord_list()

    ds_minr = ds.dataset_dict["rewards"].min()
    print(f"Dataset minimum reward = {ds_minr}")
    print("observation shape:", env.observation_space.sample().shape)
    print("action shape:", env.action_space.sample().shape)

    record_step = 0
    if FLAGS.jsrl_ratio > 0.0:
        bc_save_dir = f"diffusion_checkpoints/dbc-{FLAGS.env_name}/seed-{FLAGS.seed}"
        bc_save_dir = os.path.abspath(bc_save_dir)
        rng = jax.random.PRNGKey(FLAGS.seed)
        bc = DiffusionBC.create(
            FLAGS.diff_config,
            rng,
            env.observation_space.sample(),
            env.action_space.sample(),
        )
        try:
            restored_bc = orbax_checkpointer.restore(bc_save_dir, item=bc)
        except:
            restored_bc = None

        if restored_bc is None:
            for _ in tqdm.tqdm(range(FLAGS.diff_bc_steps)):
                record_step += 1
                aux = True if _ % FLAGS.log_interval == 0 else False

                batch = ds.sample(FLAGS.batch_size)
                bc, bc_info = bc.update(batch, utd_ratio=1, aux=aux)

                if aux:
                    wandb.log(prefix_metrics(bc_info, "bc"), step=record_step)

                if _ % 300000 == 0:
                    curr_rng, rng = jax.random.split(rng)
                    eval_info, _ = sample_evaluate(
                        bc, curr_rng, eval_env, FLAGS.eval_episodes
                    )
                    wandb.log(
                        prefix_metrics(eval_info, "bc-evaluation"), step=record_step
                    )

            save_args = orbax_utils.save_args_from_target(bc)
            orbax_checkpointer.save(bc_save_dir, bc, save_args=save_args)
        else:
            bc = restored_bc

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.max_steps
    )
    replay_buffer.seed(FLAGS.seed)

    ########### MODELS ###########

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space, **kwargs
    )

    if FLAGS.use_rnd_offline or FLAGS.use_rnd_online:
        kwargs = dict(FLAGS.rnd_config)
        model_cls = kwargs.pop("model_cls")
        rnd = globals()[model_cls].create(
            FLAGS.seed + 123, env.observation_space, env.action_space, **kwargs
        )
    else:
        rnd = None

    if FLAGS.offline_relabel_type == "gt":
        rm = None
    else:
        kwargs = dict(FLAGS.rm_config)
        model_cls = kwargs.pop("model_cls")
        rm = globals()[model_cls].create(
            FLAGS.seed + 123, env.observation_space, env.action_space, **kwargs
        )

    observation, done = env.reset(), False
    online_trajs = []
    online_traj = [observation]

    env_step = 0

    rng = jax.random.key(FLAGS.seed)
    if FLAGS.jsrl_ratio > 0.0:
        curr_rng, rng = jax.random.split(rng)
        rollin_enabled = (
            True if jax.random.uniform(key=curr_rng) < FLAGS.jsrl_ratio else False
        )
    else:
        rollin_enabled = False

    for i in tqdm.tqdm(
        range(0, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
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
        if "antmaze" in FLAGS.env_name:
            reward -= 1  # antmaze works better with -1/0 rewards
        env_step += 1

        online_traj.append(next_observation)

        timelimit_stop = "TimeLimit.truncated" in info

        if not done or timelimit_stop:
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

        if i >= FLAGS.start_training:
            online_batch_size = int(
                FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio)
            )
            online_batch = replay_buffer.sample(online_batch_size)
            online_batch = online_batch.unfreeze()

            if FLAGS.use_rnd_online:
                online_rnd_reward = rnd.get_reward(
                    online_batch["observations"], online_batch["actions"]
                )
                online_batch["rewards"] += online_rnd_reward

            batch = online_batch

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
                    offline_batch["rewards"][:] = ds_minr
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
                    offline_batch["rewards"] = (
                        offline_batch["rewards"] + offline_rnd_reward
                    )

                batch = combine(offline_batch, batch)

            agent, update_info = agent.update(batch, FLAGS.utd_ratio)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log(add_prefix("agent/", {k: v}), step=record_step)

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

            if rm is not None:
                rm, rm_update_info = rm.update(online_batch, FLAGS.utd_ratio)

            if rm is not None and FLAGS.offline_ratio > 0:
                rm_update_info.update(rm.evaluate(offline_batch))

            if i % FLAGS.log_interval == 0:
                if rm is not None:
                    for k, v in rm_update_info.items():
                        wandb.log(add_prefix("rm/", {k: v}), step=record_step)

        if i >= 2 * FLAGS.start_training and rnd is not None:

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
                if rnd is not None:
                    for k, v in rnd_update_info.items():
                        wandb.log(add_prefix("rnd/", {k: v}), step=record_step)

        if i % FLAGS.log_interval == 0:
            wandb.log({"env_step": env_step}, step=record_step)

        observation = next_observation

        if done:
            online_trajs.append({"observation": np.stack(online_traj, axis=0)})
            observation, done = env.reset(), False
            online_traj = [observation]
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log(add_prefix("episode/", {decode[k]: v}), step=record_step)

            if FLAGS.jsrl_ratio > 0.0:
                curr_rng, rng = jax.random.split(rng)
                rollin_enabled = (
                    True
                    if jax.random.uniform(key=curr_rng) < FLAGS.jsrl_ratio
                    else False
                )

        if i % FLAGS.eval_interval == 0:
            if "antmaze" in FLAGS.env_name:
                if rnd is not None:
                    offline_batch_size = int(
                        FLAGS.batch_size * FLAGS.utd_ratio * FLAGS.offline_ratio
                    )
                    offline_batch = ds.sample(offline_batch_size)
                    rnd_reward_plot = wandb.Image(
                        plot_rnd_reward(viz_env, offline_batch, rnd)
                    )
                    wandb.log(
                        {f"visualize/rnd_reward_plot": rnd_reward_plot},
                        step=record_step,
                    )

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

            if "antmaze" in FLAGS.env_name:

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
                wandb.log(
                    {f"visualize/offline_data_directions": image}, step=record_step
                )


if __name__ == "__main__":
    app.run(main)
