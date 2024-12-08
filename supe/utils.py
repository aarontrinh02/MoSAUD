from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict

from supe.visualization import plot_points


def list_2d_of_dicts_to_dict_of_2d_lists(list_of_dicts):
    """
    Convert a 2d list of dictionaries to a dictionary of 2d lists.
    """
    return {
        key: np.array(
            [
                [list_of_dicts[i][j][key] for j in range(len(list_of_dicts[0]))]
                for i in range(len(list_of_dicts))
            ]
        )
        for key in list_of_dicts[0][0].keys()
    }


def list_1d_of_dicts_to_dict_of_1d_lists(list_of_dicts):
    """
    Convert a 1d list of dictionaries to a dictionary of 1d lists.
    """
    return {
        key: np.array([list_of_dicts[i][key] for i in range(len(list_of_dicts))])
        for key in list_of_dicts[0].keys()
    }


def get_observation_at_index_in_chunk(observations, index):
    if type(observations) == dict or type(observations) == FrozenDict:
        return {k: v[:, index, ...] for k, v in observations.items()}
    else:
        return observations[:, index, ...]


def get_prior_statistics(agent, ds, batch_size=4096, skill_dim=8):
    index = 0
    means = np.empty((ds.dataset_dict["actions"].shape[0], skill_dim))
    stds = np.empty((ds.dataset_dict["actions"].shape[0], skill_dim))
    while index * batch_size < len(ds):
        batch = ds.sample(
            batch_size,
            indx=np.arange(index * batch_size, min((index + 1) * batch_size, len(ds))),
        )
        prior = agent.prior_model(batch["observations"])
        means[index * batch_size : (index + 1) * batch_size] = prior.loc
        stds[index * batch_size : (index + 1) * batch_size] = prior.scale_diag
        index += 1
    means, stds = np.mean(means), np.mean(stds)
    return means, stds


def view_data_distribution(viz_env, ds):
    offline_batch = ds.sample(3000)
    if (
        type(offline_batch["observations"]) == dict
        or type(offline_batch["observations"]) == FrozenDict
    ):
        vobs = offline_batch["observations"]["position"]
    else:
        vobs = offline_batch["observations"]
    return plot_points(viz_env, vobs[:, 0], vobs[:, 1])


@jax.jit
def combine(one_dict, other_dict):
    def combine_inner(v, other_v):
        tmp = jnp.empty((v.shape[0] + other_v.shape[0], *v.shape[1:]), dtype=v.dtype)
        tmp = tmp.at[0::2].set(v)
        tmp = tmp.at[1::2].set(other_v)
        return tmp

    combined = {}
    for k, v in one_dict.items():
        if isinstance(v, FrozenDict):
            combined[k] = combine(v, other_dict[k])
        else:
            combined[k] = combine_inner(v, other_dict[k])

    return combined


def add_prefix(prefix, dict):
    return {prefix + k: v for k, v in dict.items()}


@partial(jax.jit, static_argnames=("R",))
def check_overlap(coord, observations, R):
    if type(observations) == dict or type(observations) == FrozenDict:
        return jnp.any(jnp.all(jnp.abs(coord - observations["position"]) <= R, axis=-1))
    else:
        return jnp.any(jnp.all(jnp.abs(coord - observations[:, :2]) <= R, axis=-1))


def color_maze_and_configure_camera(env):
    # Update colors
    l = len(env.model.tex_type)
    sx, sy, ex, ey = 15, 45, 55, 100
    for i in range(l):
        if env.model.tex_type[i] == 0:
            height = env.model.tex_height[i]
            width = env.model.tex_width[i]
            s = env.model.tex_adr[i]
            for x in range(height):
                for y in range(width):
                    cur_s = s + (x * width + y) * 3
                    R = 192
                    r = int((ex - x) / (ex - sx) * R)
                    g = int((y - sy) / (ey - sy) * R)
                    r = np.clip(r, 0, R)
                    g = np.clip(g, 0, R)
                    env.model.tex_rgb[cur_s : cur_s + 3] = [r, g, 128]
    env.model.mat_texrepeat[0, :] = 1

    # Configure camera
    env.render(mode="rgb_array", width=200, height=200)
    env.viewer.cam.azimuth = 90.0
    env.viewer.cam.distance = 6
    env.viewer.cam.elevation = -60

    return env


def tft(observation):
    if isinstance(observation, dict):
        return {
            "pixels": observation["pixels"][..., 0],
            "state": observation["state"],
            "position": observation["position"],
        }
    else:
        return observation
