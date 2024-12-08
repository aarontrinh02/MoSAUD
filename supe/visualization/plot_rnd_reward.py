import sys

sys.path.append("../")

import matplotlib.pyplot as plt
import numpy as np
from flax.core.frozen_dict import FrozenDict
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import ImageGrid

from .visualize import get_canvas_image


def plot_rnd_reward(env, offline_batch, rnd):
    fig = plt.figure(tight_layout=True)
    axs = ImageGrid(
        fig,
        111,
        nrows_ncols=(1, 1),
        cbar_location="right",
        cbar_mode="single",
        cbar_size="5%",
        cbar_pad=0.05,
    )
    canvas = FigureCanvas(fig)

    axs[0].set_title("RND Plot")
    axs[0].axis("off")
    axs[0].set_box_aspect(1)
    env.draw(axs[0])

    if (
        type(offline_batch["observations"]) == dict
        or type(offline_batch["observations"]) == FrozenDict
    ):
        obs = offline_batch["observations"]["position"]
    else:
        obs = offline_batch["observations"]

    if (
        type(offline_batch["observations"]) == dict
        or type(offline_batch["observations"]) == FrozenDict
    ):
        rnd_values = rnd.get_reward(offline_batch)
    else:
        actions = offline_batch["actions"]
        rnd_values = rnd.get_reward(obs, actions)

    x, y = obs[:, 0], obs[:, 1]
    scatter = axs[0].scatter(
        x, y, c=rnd_values, **dict(alpha=0.75, s=5, cmap="viridis", marker="o")
    )

    axs[-1].cax.colorbar(scatter, label="Env Steps $\\left(\\times 10^3\\right)$")
    axs[-1].cax.toggle_label(True)

    image = get_canvas_image(canvas)
    plt.close()
    return image


def plot_q_values(env, offline_batch, agent):
    fig = plt.figure(tight_layout=True)
    axs = ImageGrid(
        fig,
        111,
        nrows_ncols=(1, 1),
        cbar_location="right",
        cbar_mode="single",
        cbar_size="5%",
        cbar_pad=0.05,
    )
    canvas = FigureCanvas(fig)

    axs[0].set_title("Q Values Plot")
    axs[0].axis("off")
    axs[0].set_box_aspect(1)
    env.draw(axs[0])

    if (
        type(offline_batch["observations"]) == dict
        or type(offline_batch["observations"]) == FrozenDict
    ):
        obs = offline_batch["observations"]["position"]
    else:
        obs = offline_batch["observations"]

    if (
        type(offline_batch["observations"]) == dict
        or type(offline_batch["observations"]) == FrozenDict
    ):
        rnd_values = agent.get_q(
            offline_batch["observations"], offline_batch["actions"]
        )
    else:
        actions = offline_batch["actions"]
        rnd_values = agent.get_q(obs, actions)
    x, y = obs[:, 0], obs[:, 1]
    scatter = axs[0].scatter(
        x, y, c=rnd_values, **dict(alpha=0.75, s=5, cmap="viridis", marker="o")
    )

    axs[-1].cax.colorbar(scatter, label="Env Steps $\\left(\\times 10^3\\right)$")
    axs[-1].cax.toggle_label(True)

    image = get_canvas_image(canvas)
    plt.close()
    return image
