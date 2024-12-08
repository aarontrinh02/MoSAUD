import flax.linen as nn
import jax.numpy as jnp


class StateActionValue(nn.Module):
    base_cls: nn.Module
    default_init: nn.initializers.Initializer = nn.initializers.xavier_uniform

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)

        value = nn.Dense(1, kernel_init=self.default_init())(outputs)

        return jnp.squeeze(value, -1)


class StateActionFeature(nn.Module):
    base_cls: nn.Module
    feature_dim: int
    default_init: nn.initializers.Initializer = nn.initializers.xavier_uniform

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)

        feature = nn.Dense(self.feature_dim, kernel_init=self.default_init())(outputs)

        return feature


class StateValue(nn.Module):
    base_cls: nn.Module
    default_init: nn.initializers.Initializer = nn.initializers.xavier_uniform

    @nn.compact
    def __call__(self, observations: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        inputs = observations
        outputs = self.base_cls()(inputs, *args, **kwargs)

        value = nn.Dense(1, kernel_init=self.default_init())(outputs)

        return jnp.squeeze(value, -1)


class StateFeature(nn.Module):
    base_cls: nn.Module
    feature_dim: int
    default_init: nn.initializers.Initializer = nn.initializers.xavier_uniform

    @nn.compact
    def __call__(self, observations: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        inputs = observations
        outputs = self.base_cls()(inputs, *args, **kwargs)

        feature = nn.Dense(self.feature_dim, kernel_init=self.default_init())(outputs)

        return feature
