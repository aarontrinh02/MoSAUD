import collections

import gym
import jax
import jax.numpy as jnp
import numpy as np

from supe.pretraining.opal import OPAL
from supe.utils import (get_observation_at_index_in_chunk,
                        list_1d_of_dicts_to_dict_of_1d_lists,
                        list_2d_of_dicts_to_dict_of_2d_lists, tft)


class MetaPolicyActionWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        policy: OPAL,
        rng: jax.random.PRNGKey,
        horizon: int = 4,
        eval: bool = False,
        subtract_one: bool = True,
        hilp: bool = False,
        skill_dim: int = 8,
    ):
        super().__init__(env)
        self.rng = rng
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(skill_dim,))
        self.policy = policy
        self.horizon = horizon
        self.hilp = hilp
        self.eval = eval
        self.hilp = hilp
        self.subtract_one = subtract_one
        self.observations = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.next_observation_buffer = []
        self.observation_buffer = []
        self.next_observation_buffer = []
        self.action_buffer = []
        self.data_buffer = []

    def process_buffer(self):
        if len(self.observation_buffer) > 0:
            if (
                type(self.observation_buffer[0][0]) == dict
                or type(self.observation_buffer[0][0]) == collections.OrderedDict
            ):
                self.observation_buffer = list_2d_of_dicts_to_dict_of_2d_lists(
                    self.observation_buffer
                )
            else:
                self.observation_buffer = np.array(self.observation_buffer)

            if (
                type(self.next_observation_buffer[0]) == dict
                or type(self.next_observation_buffer[0]) == collections.OrderedDict
            ):
                self.next_observation_buffer = list_1d_of_dicts_to_dict_of_1d_lists(
                    self.next_observation_buffer
                )
            else:
                self.next_observation_buffer = np.array(self.next_observation_buffer)

            if self.hilp:
                curr = get_observation_at_index_in_chunk(
                    self.observation_buffer, index=0
                )  # {key: self.observation_buffer[key][:, 0] for key in self.observation_buffer}
                curr_phi = self.policy.get_phi(tft(curr))
                next_phi = self.policy.get_phi(tft(self.next_observation_buffer))
                skills = next_phi - curr_phi
                skills = skills / np.linalg.norm(skills, axis=-1, keepdims=True)
            else:
                skills = self.policy.vae(
                    self.policy.train_state.params,
                    self.observation_buffer,
                    np.array(self.action_buffer),
                    method="encode",
                )

            results = []
            for i in range(len(self.data_buffer)):
                results.append(
                    (
                        self.data_buffer[i][0],  # observation
                        self.data_buffer[i][1],  # next observation
                        self.data_buffer[i][2],  # reward
                        self.data_buffer[i][3],  # done
                        self.data_buffer[i][4],  # mask
                        self.data_buffer[i][5],  # info
                        skills[i],  # interpolated latent action
                    )
                )
            self.next_observation_buffer = []
            self.observation_buffer = []
            self.action_buffer = []
            self.data_buffer = []
            return results
        else:
            return np.array([])

    def step(self, original_skill):
        for i in range(self.horizon):
            self.rng, curr_rng = jax.random.split(self.rng)
            if self.hilp:
                obs = tft(self.observations[-1])
                lower_action = self.policy.sample_skill_actions(
                    seed=curr_rng, observations=obs, skills=original_skill
                )
            else:
                lower_action = self.policy.sample_skill_actions(
                    rng=curr_rng,
                    observations=self.observations[-1],
                    skills=original_skill,
                )
            observation, reward, done, info = super().step(lower_action)
            self.observations.append(observation)
            self.actions.append(lower_action)
            if self.eval or not self.subtract_one:
                self.rewards.append(reward)
            else:
                self.rewards.append(reward - 1)
            self.masks.append(not done or "TimeLimit.truncated" in info)
            if done or len(self.observations) == self.horizon + 1:
                assert (
                    len(self.actions) == self.horizon
                )  # should keep last horizon actions, rewards, masks
                assert len(self.rewards) == self.horizon
                assert len(self.masks) == self.horizon
                assert (
                    len(self.observations) == self.horizon + 1
                )  # should have horizon + 1 observations, since last is next observation
                if not self.eval:
                    if not self.hilp:
                        discount = self.policy.discount
                    else:
                        discount = 0.99
                    reward_discounts = np.power(discount, np.arange(self.horizon))
                else:
                    reward_discounts = np.ones(
                        self.horizon,
                    )

                seq_rewards = self.rewards * reward_discounts
                seq_rewards = jnp.concatenate(
                    [seq_rewards[[0]], seq_rewards[1:] * self.masks[:-1]], axis=-1
                )
                total_reward = seq_rewards.sum(axis=-1)

                # Handling what to add to buffer
                if (
                    not done and i < self.horizon - 1
                ):  # if we are not done, and haven't finished executing this skill, then we will need to interpolate effective skill over last self.horizon steps later
                    self.data_buffer.append(  # so we can easily return info for replay buffer for interpolated state
                        (
                            self.observations[0],  # observation
                            self.observations[-1],  # next observation
                            total_reward,  # reward
                            done,  # done
                            np.min(self.masks),  # info
                            info,  # mask
                        )
                    )
                    self.action_buffer.append(
                        self.actions[:]
                    )  # trailing actions for VAE to interpolate latent
                    self.observation_buffer.append(
                        self.observations[:-1]
                    )  # trailing observations for VAE to interpolate latent
                    self.next_observation_buffer.append(
                        self.observations[-1]
                    )  # next observation for HILP latent interpolation
                else:  # if we are done, or have finished executing this skill, then we want to return the total reward and next observation (which is the current one)
                    next_observation = self.observations[-1]

                # Handling deletion of old data
                if (
                    done
                ):  # if we are done, then we want to reset the buffer, since we shouldn't estimate skills across trajectories
                    self.observations = []
                    self.actions = []
                    self.rewards = []
                    self.masks = []
                    break
                else:  # otherwise, just remove the oldest observation, action, reward, and mask from the buffer
                    self.observations = self.observations[1:]
                    self.actions = self.actions[1:]
                    self.rewards = self.rewards[1:]
                    self.masks = self.masks[1:]

        return next_observation, total_reward, done, info

    def reset(self):
        observation = super().reset()
        self.observations.append(observation)
        return observation
