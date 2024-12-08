import gym

from supe.wrappers.mask_kitchen_goal import MaskKitchenGoal
from supe.wrappers.meta_env_wrapper import MetaPolicyActionWrapper
from supe.wrappers.render_observation import RenderObservation
from supe.wrappers.single_precision import SinglePrecision
from supe.wrappers.tanh_converter import TanhConverter
from supe.wrappers.universal_seed import UniversalSeed
from supe.wrappers.wandb_video import WANDBVideo


def wrap_gym(
    env: gym.Env, rescale_actions: bool = True, render_image: bool = False
) -> gym.Env:
    env = SinglePrecision(env)
    env = UniversalSeed(env)
    if rescale_actions:
        env = gym.wrappers.RescaleAction(
            env, -1, 1
        )  # assures error thrown if action is out of bounds

    if render_image:
        env = RenderObservation(env)

    env = gym.wrappers.ClipAction(env)

    return env
