import gym
import numpy as np


class MaskKitchenGoal(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(30,))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs[:30], reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs[:30]
