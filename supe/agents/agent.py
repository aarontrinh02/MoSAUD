from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.training.train_state import TrainState

from supe.types import PRNGKey


@partial(jax.jit, static_argnames="apply_fn")
def _sample_actions(rng, apply_fn, params, observations: np.ndarray) -> np.ndarray:
    key, rng = jax.random.split(rng)
    dist = apply_fn({"params": params}, observations)
    return dist.sample(seed=key), rng


@partial(jax.jit, static_argnames="apply_fn")
def _eval_actions(apply_fn, params, observations: np.ndarray) -> np.ndarray:
    dist = apply_fn({"params": params}, observations)
    return dist.mode()


class Agent(struct.PyTreeNode):
    actor: TrainState
    rng: PRNGKey

    def sample_actions(self, observations: np.ndarray) -> Tuple[np.ndarray, "Agent"]:
        actions, new_rng = _sample_actions(self.rng, self.actor.apply_fn, self.actor.params, observations)
        return np.asarray(actions), self.replace(rng=new_rng)

    @partial(jax.jit, static_argnames=('self'))
    def sample_actions_jit(self, observations: jnp.ndarray) -> Tuple[jnp.ndarray, "Agent"]:
        dist = self.actor.apply_fn({"params": self.actor.params}, observations)
        key, new_rng = jax.random.split(self.rng)
        return dist.sample(seed=key), self.replace(rng=new_rng)

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = _eval_actions(self.actor.apply_fn, self.actor.params, observations)
        return np.asarray(actions)

    @partial(jax.jit, static_argnames=('self'))
    def eval_actions_jit(self, observations: jnp.ndarray) -> jnp.ndarray:
        dist = self.actor.apply_fn({"params": self.actor.params}, observations)
        return dist.mode()

    @jax.jit
    def sample(self, observations):
        dist = self.actor.apply_fn({"params": self.actor.params}, observations)
        key, rng = jax.random.split(self.rng)
        actions = dist.sample(seed=key)
        return actions, self.replace(rng=rng)
