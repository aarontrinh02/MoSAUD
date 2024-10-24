import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict


class TanhConverter:
    def to_tanh(self, latent, epsilon=1e-5):
        return jnp.clip(jnp.tanh(latent), -1 + epsilon, 1 - epsilon)

    def from_tanh(self, latent, epsilon=1e-5):
        return jnp.arctanh(jnp.clip(latent, -1 + epsilon, 1 - epsilon))
