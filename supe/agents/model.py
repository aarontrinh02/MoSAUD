import functools
from typing import Any, Callable, Optional, Sequence, Type

import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow_probability


def default_init(scale: Optional[float] = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


def orthogonal_init(scale: Optional[float] = jnp.sqrt(2.0)):
    return jax.nn.initializers.orthogonal(scale)


INIT_FNS = {
    None: default_init,
    "orthogonal": orthogonal_init,
}

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors


def update_target_network(main_params, target_params, tau):
    return jax.tree_util.tree_map(
        lambda x, y: tau * x + (1.0 - tau) * y, main_params, target_params
    )


def value_and_multi_grad(fun, n_outputs, argnums=0):
    def select_output(index):
        def wrapped(*args, **kwargs):
            x, *aux = fun(*args, **kwargs)
            return (x[index], *aux)

        return wrapped

    grad_fns = tuple(
        jax.value_and_grad(select_output(i), argnums=argnums, has_aux=True)
        for i in range(n_outputs)
    )

    def multi_grad_fn(*args, **kwargs):
        grads, values = [], []
        for grad_fn in grad_fns:
            (value, *aux), grad = grad_fn(*args, **kwargs)
            values.append(value)
            grads.append(grad)
        return (tuple(values), *aux), tuple(grads)

    return multi_grad_fn


def broadcast_concatenate(*arrs):
    shape = jnp.broadcast_shapes(*map(lambda x: x.shape[:-1], arrs))
    return jnp.concatenate(
        tuple(map(lambda x: jnp.broadcast_to(x, shape=shape + (x.shape[-1],)), arrs)),
        axis=-1,
    )


"""
Both classes below are taken from the link below (note that the initialization used originally are xavier_uniform)
https://github.com/philippe-eecs/IDQL/blob/main/jaxrl5/networks/resnet.py#L32
"""


class MLPResNetBlock(nn.Module):
    """MLPResNet block."""

    features: int
    act: Callable
    dropout_rate: float = None
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x, training: bool = False):
        residual = x
        if self.dropout_rate is not None and self.dropout_rate > 0.0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.Dense(self.features * 4)(x)
        x = self.act(x)
        x = nn.Dense(self.features)(x)

        if residual.shape != x.shape:
            residual = nn.Dense(self.features)(residual)

        return residual + x


class MLPResNet(nn.Module):
    num_blocks: int
    out_dim: int
    dropout_rate: float = None
    use_layer_norm: bool = False
    hidden_dim: int = 256
    activations: Callable = nn.relu
    kernel_init_type: Optional[str] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        init_fn = INIT_FNS[self.kernel_init_type]
        x = nn.Dense(self.hidden_dim, kernel_init=init_fn())(x)
        for _ in range(self.num_blocks):
            x = MLPResNetBlock(
                self.hidden_dim,
                act=self.activations,
                use_layer_norm=self.use_layer_norm,
                dropout_rate=self.dropout_rate,
            )(x, training=training)

        x = self.activations(x)
        x = nn.Dense(self.out_dim, kernel_init=init_fn())(x)
        return x


class Ensemble(nn.Module):
    net_cls: Type[nn.Module]
    num: int = 2

    @nn.compact
    def __call__(self, *args, **kwargs):
        ensemble = nn.vmap(
            self.net_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num,
        )
        return ensemble()(*args, **kwargs)


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    use_layer_norm: bool = False
    scale_final: Optional[float] = None
    dropout_rate: Optional[float] = None
    use_pnorm: bool = False
    kernel_init_type: Optional[str] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: float = False) -> jnp.ndarray:

        for i, size in enumerate(self.hidden_dims):
            # init_fn = orthogonal_init if self.orthogonal_init else default_init
            init_fn = INIT_FNS[self.kernel_init_type]
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size, kernel_init=init_fn(self.scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=init_fn())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        if self.use_pnorm:
            x /= jnp.linalg.norm(x, axis=-1, keepdims=True).clip(1e-10)
        return x


class StateActionValue(nn.Module):
    base_cls: nn.Module
    kernel_init_type: Optional[str] = None
    kernel_init_scale: float = 1.0

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, *args, **kwargs
    ) -> jnp.ndarray:
        # inputs = jnp.concatenate([observations, actions], axis=-1)
        inputs = broadcast_concatenate(observations, actions)
        outputs = self.base_cls()(inputs, *args, **kwargs)

        # init_fn = orthogonal_init if self.orthogonal_init else default_init
        init_fn = INIT_FNS[self.kernel_init_type]
        value = nn.Dense(1, kernel_init=init_fn(self.kernel_init_scale))(outputs)

        return jnp.squeeze(value, -1)


class StateValue(nn.Module):
    base_cls: nn.Module
    kernel_init_type: Optional[str] = None
    kernel_init_scale: float = 1.0

    @nn.compact
    def __call__(self, observations: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        outputs = self.base_cls()(observations, *args, **kwargs)

        # init_fn = orthogonal_init if self.orthogonal_init else default_init
        init_fn = INIT_FNS[self.kernel_init_type]
        value = nn.Dense(1, kernel_init=init_fn(self.kernel_init_scale))(outputs)

        return jnp.squeeze(value, -1)


class TanhTransformedDistribution(tfd.TransformedDistribution):
    def __init__(self, distribution: tfd.Distribution, validate_args: bool = False):
        super().__init__(
            distribution=distribution, bijector=tfb.Tanh(), validate_args=validate_args
        )

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

    def sample_and_log_prob(self, *args, **kwargs):
        x = self.sample(*args, **kwargs)
        return x, self.log_prob(x)

    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties


class Normal(nn.Module):
    base_cls: Type[nn.Module]
    action_dim: int
    fixed_log_std: bool = False
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    state_dependent_std: bool = True
    squash_tanh: bool = False
    learnable_log_std_multiplier: Optional[float] = None  # learnable log_std multiplier
    learnable_log_std_offset: Optional[float] = None  # learnable log_std offset
    kernel_init_scale: float = 1.0
    kernel_init_type: Optional[str] = None

    @nn.compact
    def __call__(self, inputs, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(inputs, *args, **kwargs)

        init_fn = INIT_FNS[
            self.kernel_init_type
        ]  # orthogonal_init if self.orthogonal_init else default_init
        means = nn.Dense(
            self.action_dim,
            kernel_init=init_fn(self.kernel_init_scale),
            name="OutputDenseMean",
        )(x)

        if self.fixed_log_std:
            log_stds = jnp.zeros_like(means)
        else:
            if self.state_dependent_std:
                log_stds = nn.Dense(
                    self.action_dim,
                    kernel_init=init_fn(self.kernel_init_scale),
                    name="OutputDenseLogStd",
                )(x)
            else:
                log_stds = self.param(
                    "OutpuLogStd",
                    nn.initializers.zeros,
                    (self.action_dim,),
                    jnp.float32,
                )

            if self.learnable_log_std_multiplier is not None:
                log_stds *= self.param(
                    "LogStdMul",
                    nn.initializers.constant(self.learnable_log_std_multiplier),
                    (),
                    jnp.float32,
                )
            if self.learnable_log_std_offset is not None:
                log_stds += self.param(
                    "LogStdOffset",
                    nn.initializers.constant(self.learnable_log_std_offset),
                    (),
                    jnp.float32,
                )

            log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        distribution = tfd.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds)
        )

        if self.squash_tanh:
            return TanhTransformedDistribution(distribution)
        else:
            return distribution


TanhNormal = functools.partial(Normal, squash_tanh=True)
