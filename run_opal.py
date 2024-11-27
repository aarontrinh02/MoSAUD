import os

import absl.app
import absl.flags
import gym
import jax
jax.config.update('jax_platform_name', 'cpu')
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from absl import flags
from flax.training import checkpoints
from gym import spaces
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from ml_collections import config_flags
import wandb

from supe.agents.drq.augmentations import batched_random_crop
from supe.data import ChunkDataset, D4RLDataset, Dataset
from supe.pretraining.opal import OPAL
from supe.utils import color_maze_and_configure_camera, view_data_distribution
from supe.visualization import (get_canvas_image, get_env_and_dataset,
                                plot_trajectories)
from supe.wrappers import wrap_gym
from supe.wrappers.mask_kitchen_goal import MaskKitchenGoal

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "antmaze-large-diverse-v2", "Name of the environment")
flags.DEFINE_integer("seed", 1, "Random seed")
flags.DEFINE_integer("log_period", 10000, "Logging period")
flags.DEFINE_integer("eval_period", 100000, "Evaluation period")
flags.DEFINE_integer("save_period", 100000, "Model saving period")
flags.DEFINE_integer("num_eval_trajectories", 10, "Number of evaluation trajectories")
flags.DEFINE_integer("max_steps", 1000000, "Maximum number of steps")
flags.DEFINE_integer("batch_size", 256, "Batch size")
flags.DEFINE_integer("horizon_length", 4, "Horizon length")
flags.DEFINE_string(
    "save_dir",
    "./opal_checkpoints",
    "Root directory to save checkpoints",
)
flags.DEFINE_string("project_name", "opal", "Name of the wandb project")
flags.DEFINE_boolean("debug", False, "Enable debug mode")
flags.DEFINE_boolean("vision", False, "Enable vision")

config_flags.DEFINE_config_file(
    "config",
    "configs/opal_config.py",
    "File path to the opal hyperparameter configuration.",
    lock_config=False,
)


def rollout_skill_agent(agent, horizon, env, vision=False, rng=None):
    observation, done = env.reset(), False
    if vision:
        pos = observation["position"]
    else:
        pos = observation[:2]

    positions = [pos]
    rewards = []

    i = 0
    skill = None
    while not done:
        if i % horizon == 0:
            if rng is not None:
                rng, curr_rng = jax.random.split(rng)
                skill = agent.sample_skills(rng=curr_rng, observations=observation)
            else:
                skill = agent.eval_skills(observations=observation)

        if rng is not None:
            rng, curr_rng = jax.random.split(rng)
            action = agent.sample_skill_actions(
                rng=curr_rng, observations=observation, skills=skill
            )
        else:
            action = agent.eval_skill_actions(observations=observation, skills=skill)

        observation, reward, done, info = env.step(action)

        if FLAGS.vision:
            positions.append(observation["position"])
        else:
            positions.append(observation[:2])

        rewards.append(reward)
        i += 1

    return {
        "observation": np.stack(
            positions, axis=0
        ),  # we only care about 2D positions for plotting trajectories
        "return": np.sum(rewards),
        "length": len(rewards),
    }


def main(_):

    FLAGS = absl.flags.FLAGS

    wandb.init(project=FLAGS.project_name)
    wandb.config.update(FLAGS)

    if FLAGS.debug:
        FLAGS.max_steps = 1000
        FLAGS.num_eval_trajectories = 1
        FLAGS.eval_period = 500
        FLAGS.log_period = 10
        FLAGS.checkpoint_model = False
        FLAGS.checkpoint_buffer = False

    env = wrap_gym(
        gym.make(FLAGS.env_name), rescale_actions=True, render_image=FLAGS.vision
    )
    eval_env = wrap_gym(
        gym.make(FLAGS.env_name), rescale_actions=True, render_image=FLAGS.vision
    )

    if "kitchen" in FLAGS.env_name:
        env = MaskKitchenGoal(env)
        env.env.env.env.env.env.env.env.REMOVE_TASKS_WHEN_COMPLETE = False
        eval_env = MaskKitchenGoal(eval_env)

    observation_space, action_space = eval_env.observation_space, eval_env.action_space
    if FLAGS.vision:
        observation_space = gym.spaces.Dict(
            {
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(27,)),
                "pixels": spaces.Box(
                    low=0, high=255, shape=(64, 64, 3, 1), dtype=np.uint8
                ),
                "position": spaces.Box(low=-np.inf, high=np.inf, shape=(2,)),
            }
        )

    observations, actions = observation_space.sample(), action_space.sample()

    if "antmaze" in FLAGS.env_name:

        viz_env, viz_dataset = get_env_and_dataset(FLAGS.env_name)

    rng = jax.random.PRNGKey(FLAGS.seed)

    agent_rng, rng = jax.random.split(rng)
    agent = OPAL.create(
        FLAGS.config,
        agent_rng,
        observations,
        actions,
        chunk_size=FLAGS.horizon_length,
        cnn=FLAGS.vision,
    )

    dataset = D4RLDataset(
        env,
        subtract_one="antmaze" in FLAGS.env_name,
        remove_kitchen_goal="kitchen" in FLAGS.env_name,
    )

    if FLAGS.vision:
        env = color_maze_and_configure_camera(env)
        eval_env = color_maze_and_configure_camera(eval_env)
        image_dataset = dict(np.load(f"data/antmaze_topview_6_60/{FLAGS.env_name}.npz"))

        dataset.dataset_dict["observations"] = dict(
            position=dataset.dataset_dict["observations"][:, :2],
            state=dataset.dataset_dict["observations"][:, 2:],
            pixels=image_dataset["images"],
        )

        dataset.dataset_dict["next_observations"] = dict(
            position=dataset.dataset_dict["next_observations"][:, :2],
            state=dataset.dataset_dict["next_observations"][:, 2:],
            pixels=image_dataset["next_images"],
        )

    dataset = ChunkDataset.create(
        dataset=dataset,
        chunk_size=FLAGS.horizon_length,
        agent=None,
        tanh_converter=None,
        label_skills=False,
    )

    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, dynamic_ncols=True
    ):
        curr_rng, rng = jax.random.split(rng)
        batch = dataset.sample_chunk((FLAGS.batch_size,), rng=curr_rng)

        if FLAGS.vision:
            rng, curr_rng = jax.random.split(rng)
            batch["seq_observations"] = batched_random_crop(
                curr_rng,
                batch["seq_observations"],
                "pixels",
                frozen=True,
            )
            rng, curr_rng = jax.random.split(rng)
            batch["next_seq_observations"] = batched_random_crop(
                curr_rng,
                batch["next_seq_observations"],
                "pixels",
                frozen=True,
            )

        logging = i % FLAGS.log_period == 0 or i == 1
        agent, vae_info = agent.update_vae(batch, aux=logging)
        agent, iql_info = agent.update_iql(batch, aux=logging)

        if logging:
            train_metrics = {
                f"training/{k}": v for k, v in {**vae_info, **iql_info}.items()
            }
            for k, v in train_metrics.items():
                wandb.log({k: v}, step=i)

        if i % FLAGS.eval_period == 0 or i == 1:

            curr_rng, rng = jax.random.split(rng)
            sample_trajs, eval_trajs = [], []
            rngs = jax.random.split(curr_rng, FLAGS.num_eval_trajectories)
            for curr_rng in rngs:
                eval_trajs.append(
                    rollout_skill_agent(
                        agent,
                        FLAGS.horizon_length,
                        eval_env,
                        rng=None,
                        vision=FLAGS.vision,
                    )
                )  # deterministic when no rng is provided
                sample_trajs.append(
                    rollout_skill_agent(
                        agent,
                        FLAGS.horizon_length,
                        eval_env,
                        rng=curr_rng,
                        vision=FLAGS.vision,
                    )
                )  # sampling from both skills and h-agent when rng is provided

            eval_metrics = {
                "evaluation/deterministic-return": np.mean(
                    [traj["return"] for traj in eval_trajs]
                ),
                "evaluation/deterministic-length": np.mean(
                    [traj["length"] for traj in eval_trajs]
                ),
                "evaluation/sample-return": np.mean(
                    [traj["return"] for traj in sample_trajs]
                ),
                "evaluation/sample-length": np.mean(
                    [traj["length"] for traj in sample_trajs]
                ),
            }

            for k, v in eval_metrics.items():
                wandb.log({k: v}, step=i)

            if "antmaze" in FLAGS.env_name:

                fig = plt.figure(tight_layout=True, figsize=(4, 4), dpi=200)
                canvas = FigureCanvas(fig)
                plot_trajectories(viz_env, viz_dataset, eval_trajs, fig, plt.gca())
                image = wandb.Image(get_canvas_image(canvas))
                wandb.log({f"visualize/trajs": image}, step=i)
                plt.close(fig)

                data_distribution_im = view_data_distribution(viz_env, dataset)
                image = wandb.Image(data_distribution_im)
                wandb.log({f"visualize/offline_data_dist": image}, step=i)

        if i == 1 or (i % FLAGS.save_period == 0 and FLAGS.save_dir is not None):
            os.makedirs(FLAGS.save_dir, exist_ok=True)
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.save_dir)
                + "/"
                + str(FLAGS.env_name)
                + "/vision="
                + str(FLAGS.vision)
                + "/horizon="
                + str(FLAGS.horizon_length)
                + "/seed="
                + str(FLAGS.seed),
                agent,
                i,
                keep=100,
                overwrite=True,
            )


if __name__ == "__main__":
    absl.app.run(main)
