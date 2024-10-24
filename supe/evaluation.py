from typing import Dict

import gym
import jax
import numpy as np

from supe.wrappers import TanhConverter
from supe.wrappers.wandb_video import WANDBVideo


def evaluate(
    agent,
    env: gym.Env,
    num_episodes: int,
    tanh_converter: TanhConverter = None,
    save_video: bool = False,
    save_video_name="eval_video",
    hilp=False,
) -> Dict[str, float]:

    if save_video:
        env = WANDBVideo(env, name=save_video_name, max_videos=1)

    trajs = []
    cum_returns = []
    cum_lengths = []
    for i in range(num_episodes):
        observation, done = env.reset(), False
        traj = [observation]
        cum_return = 0
        cum_length = 0
        while not done:
            action = agent.eval_actions(observation)
            if tanh_converter is not None:
                action = tanh_converter.from_tanh(action)
            if hilp:
                action = action / np.linalg.norm(action)
            observation, reward, done, info = env.step(action)
            done = done or "TimeLimit.truncated" in info
            cum_return += reward
            cum_length += 1
            traj.append(observation)
        cum_returns.append(cum_return)
        cum_lengths.append(cum_length)
        trajs.append({"observation": np.stack(traj, axis=0)})
    return {"return": np.mean(cum_returns), "length": np.mean(cum_lengths)}, trajs


def sample_evaluate(
    agent,
    rng,
    env: gym.Env,
    num_episodes: int,
    save_video: bool = False,
    save_video_name="eval_video",
) -> Dict[str, float]:

    if save_video:
        env = WANDBVideo(env, name=save_video_name, max_videos=1)

    trajs = []
    cum_returns = []
    cum_lengths = []
    for i in range(num_episodes):
        observation, done = env.reset(), False
        traj = [observation]
        cum_return = 0
        cum_length = 0
        while not done:
            curr_rng, rng = jax.random.split(rng)
            action = agent.sample_actions(rng, observation)
            observation, reward, done, info = env.step(action)
            done = done or "TimeLimit.truncated" in info
            cum_return += reward
            cum_length += 1
            traj.append(observation)
        cum_returns.append(cum_return)
        cum_lengths.append(cum_length)
        trajs.append({"observation": np.stack(traj, axis=0)})
    return {"return": np.mean(cum_returns), "length": np.mean(cum_lengths)}, trajs
