import matplotlib

matplotlib.use("Agg")

import d4rl
import gym
import matplotlib.pyplot as plt
import numpy as np
from flax.core.frozen_dict import FrozenDict
from matplotlib import patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def get_canvas_image(canvas):
    canvas.draw()
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
    out_image = out_image.reshape(canvas.get_width_height()[::-1] + (3,))
    return out_image


def get_inner_env(env):
    if hasattr(env, "_maze_size_scaling"):
        return env
    elif hasattr(env, "env"):
        return get_inner_env(env.env)
    elif hasattr(env, "wrapped_env"):
        return get_inner_env(env.wrapped_env)
    return env


class GoalReachingAnt(gym.Wrapper):
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.inner_env = get_inner_env(self.env)
        self.observation_space = gym.spaces.Dict(
            {
                "observation": self.env.observation_space,
                "goal": self.env.observation_space,
            }
        )
        self.action_space = self.env.action_space

    def step(self, action):
        next_obs, r, done, info = self.env.step(action)

        achieved = self.get_xy()
        desired = self.target_goal
        distance = np.linalg.norm(achieved - desired)
        info["x"], info["y"] = achieved
        info["achieved_goal"] = np.array(achieved)
        info["desired_goal"] = np.copy(desired)
        info["success"] = float(distance < 0.5)
        done = "TimeLimit.truncated" in info

        return self.get_obs(next_obs), r, done, info

    def get_obs(self, obs):
        target_goal = obs.copy()
        target_goal[:2] = self.target_goal
        return dict(observation=obs, goal=target_goal)

    def reset(self):
        obs = self.env.reset()
        return self.get_obs(obs)

    def get_starting_boundary(self):
        self = self.inner_env
        torso_x, torso_y = self._init_torso_x, self._init_torso_y
        S = self._maze_size_scaling
        return (0 - S / 2 + S - torso_x, 0 - S / 2 + S - torso_y), (
            len(self._maze_map[0]) * S - torso_x - S / 2 - S,
            len(self._maze_map) * S - torso_y - S / 2 - S,
        )

    def XY(self, n=20):
        bl, tr = self.get_starting_boundary()
        X = np.linspace(
            bl[0] + 0.02 * (tr[0] - bl[0]), tr[0] - 0.02 * (tr[0] - bl[0]), n
        )
        Y = np.linspace(
            bl[1] + 0.02 * (tr[1] - bl[1]), tr[1] - 0.02 * (tr[1] - bl[1]), n
        )

        X, Y = np.meshgrid(X, Y)
        states = np.array([X.flatten(), Y.flatten()]).T
        return states

    def draw(self, ax=None):
        if not ax:
            ax = plt.gca()
        self = self.inner_env
        torso_x, torso_y = self._init_torso_x, self._init_torso_y
        S = self._maze_size_scaling
        for i in range(len(self._maze_map)):
            for j in range(len(self._maze_map[0])):
                struct = self._maze_map[i][j]
                if struct == 1:
                    rect = patches.Rectangle(
                        (j * S - torso_x - S / 2, i * S - torso_y - S / 2),
                        S,
                        S,
                        linewidth=1,
                        edgecolor="none",
                        facecolor="grey",
                        alpha=1.0,
                    )

                    ax.add_patch(rect)
        ax.set_xlim(
            0 - S / 2 + 0.6 * S - torso_x,
            len(self._maze_map[0]) * S - torso_x - S / 2 - S * 0.6,
        )
        ax.set_ylim(
            0 - S / 2 + 0.6 * S - torso_y,
            len(self._maze_map) * S - torso_y - S / 2 - S * 0.6,
        )

    def get_coord_list(self):
        coords = []
        self = self.inner_env
        torso_x, torso_y = self._init_torso_x, self._init_torso_y
        S = self._maze_size_scaling
        for i in range(len(self._maze_map)):
            for j in range(len(self._maze_map[0])):
                struct = self._maze_map[i][j]
                if struct != 1:
                    coords.append((j * S - torso_x, i * S - torso_y))
        return coords, S


def get_env_and_dataset(env_name):
    env = GoalReachingAnt(env_name)
    dataset = d4rl.qlearning_dataset(env)
    dataset["masks"] = 1.0 - dataset["terminals"]
    dataset["dones_float"] = 1.0 - np.isclose(
        np.roll(dataset["observations"], -1, axis=0), dataset["next_observations"]
    ).all(-1)
    return env, dataset


def plot_trajectories(env, dataset, trajectories, fig, ax, color_list=None):
    if color_list is None:
        from itertools import cycle

        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color_list = cycle(color_cycle)

    for color, trajectory in zip(color_list, trajectories):
        if type(trajectory["observation"][0]) == dict:
            obs = np.array([d["position"] for d in trajectory["observation"]])
        else:
            obs = np.array(trajectory["observation"])
        all_x = obs[:, 0]
        all_y = obs[:, 1]
        ax.scatter(all_x, all_y, s=5, c=color, alpha=0.02)
        ax.scatter(all_x[-1], all_y[-1], s=50, c=color, marker="*", alpha=0.3)

    env.draw(ax)


def plot_points(env, x, y):
    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)
    ax = plt.gca()
    env.draw(ax)
    ax.scatter(x, y)
    image = get_canvas_image(canvas)
    plt.close(fig)
    return image


def plot_data_directions(env, ds, N=20):
    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)
    ax = plt.gca()
    env.draw(ax)

    obs = env.XY(n=N)
    x, y = obs[:, 0], obs[:, 1]
    x = x.reshape(N, N)
    y = y.reshape(N, N)
    dataset_dict = ds.sample(3000)
    if (
        type(dataset_dict["observations"]) == dict
        or type(dataset_dict["observations"]) == FrozenDict
    ):
        o1 = dataset_dict["observations"]["position"]
        o2 = dataset_dict["next_observations"]["position"]
    else:
        o1 = dataset_dict["observations"][:, :2]
        o2 = dataset_dict["next_observations"][:, :2]
    delta = o2 - o1

    D = np.zeros(obs.shape)
    T = np.zeros(obs.shape)
    for o, d in zip(o1, delta):
        i = np.argmin(np.linalg.norm(o - obs, axis=-1))
        D[i] += d
        T[i] += 1
    D = np.nan_to_num(D / T)

    dx, dy = D[:, 0], D[:, 1]
    dx = dx.reshape(N, N)
    dy = dy.reshape(N, N)

    mesh = ax.quiver(x, y, dx, dy, scale=3, scale_units="width")

    image = get_canvas_image(canvas)
    plt.close(fig)
    return image
