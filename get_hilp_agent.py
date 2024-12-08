import glob
import os
import platform
import time
from datetime import datetime
from email.mime import base, image

if "mac" in platform.platform():
    pass
else:
    os.environ["MUJOCO_GL"] = "egl"
    if "SLURM_STEP_GPUS" in os.environ:
        os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_STEP_GPUS"]

import sys
from functools import partial

import flax
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags

sys.path.insert(0, os.path.abspath("hilp/hilp_gcrl"))

import pickle

from ml_collections import ConfigDict

from hilp import ant_diagnostics, d4rl_ant, d4rl_utils
from hilp.agents import hilp as learner
from jaxrl_m.evaluation import EpisodeMonitor


def get_default_config(updates=None):
    config = ConfigDict()
    config.agent_name = "hilp"
    config.env_name = "antmaze-large-diverse-v2"
    config.save_dir = "exp/"
    config.run_group = "Debug"
    config.seed = 0
    config.eval_episodes = 50
    config.num_video_episodes = 2
    config.log_interval = 1000
    config.eval_interval = 100000
    config.save_interval = 100000
    config.batch_size = 1024
    config.train_steps = 1000000
    config.lr = 3e-4
    config.value_hidden_dim = 512
    config.value_num_layers = 3
    config.actor_hidden_dim = 512
    config.actor_num_layers = 3
    config.discount = 0.99
    config.tau = 0.005
    config.expectile = 0.95
    config.use_layer_norm = 1
    config.skill_dim = 32
    config.skill_expectile = 0.9
    config.skill_temperature = 10
    config.skill_discount = 0.99
    config.p_currgoal = 0.0
    config.p_trajgoal = 0.625
    config.p_randomgoal = 0.375
    config.planning_num_recursions = 0
    config.planning_num_states = 50000
    config.planning_num_knns = 50
    config.encoder = None
    config.p_aug = None
    config.algo_name = None

    if updates is not None:
        config.update(ConfigDict(updates).copy_and_resolve_references())
    return config


def get_env_and_dataset(env_name, visual):
    aux_env = {}
    goal_info = {}
    if "antmaze" in env_name:

        import gym

        env = gym.make(env_name)
        env = EpisodeMonitor(env)

        dataset = d4rl_utils.get_dataset(env, env_name, goal_conditioned=True)
        dataset = dataset.copy({"rewards": dataset["rewards"] - 1.0})

        if visual:
            env.render(mode="rgb_array", width=200, height=200)
            if "large" in env_name:
                env.viewer.cam.lookat[0] = 18
                env.viewer.cam.lookat[1] = 12
                env.viewer.cam.distance = 50
                env.viewer.cam.elevation = -90
            elif "ultra" in env_name:
                env.viewer.cam.lookat[0] = 26
                env.viewer.cam.lookat[1] = 18
                env.viewer.cam.distance = 70
                env.viewer.cam.elevation = -90
            else:
                raise NotImplementedError
    elif "kitchen" in env_name:
        if "visual" in env_name:
            from hilp.d4rl_utils import kitchen_render

            orig_env_name = env_name.split("visual-")[1]
            env = d4rl_utils.make_env(orig_env_name)
            dataset = dict(np.load(f"data/d4rl_kitchen_rendered/{orig_env_name}.npz"))
            dataset = d4rl_utils.get_dataset(
                env, env_name, dataset=dataset, filter_terminals=True
            )

            state = env.reset()
            # Random example state from the dataset for proprioceptive states
            goal_state = [
                -2.3403780e00,
                -1.3053924e00,
                1.1021180e00,
                -1.8613019e00,
                1.5087037e-01,
                1.7687809e00,
                1.2525779e00,
                2.9698312e-02,
                3.0899283e-02,
                3.9908718e-04,
                4.9550228e-05,
                -1.9946630e-05,
                2.7519276e-05,
                4.8786267e-05,
                3.2835731e-05,
                2.6504624e-05,
                3.8422750e-05,
                -6.9888681e-01,
                -5.0150707e-02,
                3.4855098e-01,
                -9.8701166e-03,
                -7.6958216e-03,
                -8.0031347e-01,
                -1.9142720e-01,
                7.2064394e-01,
                1.6191028e00,
                1.0021452e00,
                -3.2998802e-04,
                3.7205056e-05,
                5.3616576e-02,
            ]
            goal_state[9:] = state[39:]  # Set goal object states
            env.sim.set_state(np.concatenate([goal_state, env.init_qvel]))
            env.sim.forward()
            goal_info = {
                "ob": kitchen_render(env).astype(np.float32),
            }
            env.reset()
        else:
            env = d4rl_utils.make_env(env_name)
            dataset = d4rl_utils.get_dataset(env, env_name, filter_terminals=True)
            dataset = dataset.copy(
                {
                    "observations": dataset["observations"][:, :30],
                    "next_observations": dataset["next_observations"][:, :30],
                }
            )
    else:
        raise NotImplementedError

    return env, dataset, aux_env, goal_info


def get_restore_path(env_name, base_path="hilp_checkpoints", visual=False, seed=0):

    if "mixed" in env_name:
        path = os.path.join(base_path, "Mixed")
    elif "partial" in env_name:
        path = os.path.join(base_path, "Partial")
    elif "complete" in env_name:
        path = os.path.join(base_path, "Complete")
    elif "antmaze" in env_name and visual:
        path = os.path.join(base_path, "VAM")
    elif "antmaze" in env_name and not visual:
        if len(env_name.split("-")) == 5:
            env_name = env_name[:-2]
        path = os.path.join(base_path, env_name)
    else:
        assert False, f"invalid environment name '{env_name}'"

    for dirname in os.listdir(path):
        p = os.path.join(path, dirname)
        if os.path.isdir(p) and dirname.startswith(f"sd{seed:03d}"):
            return os.path.abspath(p)


def load_hilp_agent(
    config, restore_path, image_dataset=None, restore_epoch=500000, visual=False
):
    env_name = config.env_name

    if env_name.endswith("-2") or env_name.endswith("-3") or env_name.endswith("-4"):
        env_name = env_name[:-2]

    env, dataset, _, _ = get_env_and_dataset(env_name, visual=visual)

    if image_dataset is not None:
        dataset = dataset.copy(
            {
                "observations": dict(
                    position=dataset["observations"][:, :2],
                    state=dataset["observations"][:, 2:],
                    pixels=image_dataset["images"],
                ),
                "next_observations": dict(
                    position=dataset["next_observations"][:, :2],
                    state=dataset["next_observations"][:, 2:],
                    pixels=image_dataset["next_images"],
                ),
            }
        )
    env.reset()

    example_batch = dataset.sample(1)
    agent = learner.create_learner(
        config.seed,
        example_batch["observations"],
        example_batch["actions"],
        lr=config.lr,
        value_hidden_dims=(config.value_hidden_dim,) * config.value_num_layers,
        actor_hidden_dims=(config.actor_hidden_dim,) * config.actor_num_layers,
        discount=config.discount,
        tau=config.tau,
        expectile=config.expectile,
        use_layer_norm=config.use_layer_norm,
        skill_dim=config.skill_dim,
        skill_expectile=config.skill_expectile,
        skill_temperature=config.skill_temperature,
        skill_discount=config.skill_discount,
        encoder=config.encoder,
    )

    candidates = glob.glob(restore_path)
    if len(candidates) == 0:
        raise Exception(f"Path does not exist: {restore_path}")
    if len(candidates) > 1:
        raise Exception(f"Multiple matching paths exist for: {restore_path}")
    if restore_epoch is None:
        restore_path = candidates[0] + "/params.pkl"
    else:
        restore_path = candidates[0] + f"/params_{restore_epoch}.pkl"
    with open(restore_path, "rb") as f:
        load_dict = pickle.load(f)
    agent = flax.serialization.from_state_dict(agent, load_dict["agent"])
    print(f"Restored from {restore_path}")
    return agent
