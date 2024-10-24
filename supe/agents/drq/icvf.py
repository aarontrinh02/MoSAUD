"""
adapted from the official ICVF codebase: https://github.com/dibyaghosh/icvf_release
"""

from functools import partial
from itertools import zip_longest
from typing import Callable, Dict, Sequence, Tuple

import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState

from supe.agents.drq.augmentations import batched_random_crop
from supe.data.dataset import DatasetDict
from supe.networks import MLP
from supe.networks.encoders import D4PGEncoder
from supe.networks.pixel_multiplexer import PixelMultiplexer
from supe.types import PRNGKey


class ICVF(nn.Module):
    base_cls: nn.Module
    feature_dim: int

    @nn.compact
    def __call__(self, observations: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        inputs = observations
        phi = self.base_cls(name="phi")(inputs, *args, **kwargs)
        psi = self.base_cls(name="psi")(inputs, *args, **kwargs)
        T = self.base_cls(name="T")(inputs, *args, **kwargs)
        return {
            "phi": phi,
            "psi": psi,
            "T": T,
        }


def apply_layernorm(x):
    net_def = nn.LayerNorm(use_bias=False, use_scale=False)
    return net_def.apply({"params": {}}, x)


class PixelICVF(struct.PyTreeNode):
    rng: PRNGKey
    net: TrainState
    target_net: TrainState
    data_augmentation_fn: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr: float = 3e-4,
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_filters: Sequence[int] = (3, 3, 3, 3),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        latent_dim: int = 50,
        feature_dim: int = 256,
        hidden_dims: Sequence[int] = (256, 256),
        pixel_keys: Tuple[str, ...] = ("pixels",),
        depth_keys: Tuple[str, ...] = (),
        **kwargs,
    ):
        print("Got additional kwargs: ", kwargs)

        observations = observation_space.sample()
        action_space.sample()

        rng = jax.random.PRNGKey(seed)
        rng, key1 = jax.random.split(rng, 2)

        assert cnn_features == (32, 64, 128, 256)
        pixel_encoder = D4PGEncoder(
            features=cnn_features,
            filters=cnn_filters,
            strides=cnn_strides,
            padding=cnn_padding,
        )
        rnd_base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
        )
        rnd_cls = partial(ICVF, base_cls=rnd_base_cls, feature_dim=feature_dim)
        net_def = PixelMultiplexer(
            pixel_encoder=pixel_encoder,
            network_cls=rnd_cls,
            stop_gradient=False,
            latent_dim=latent_dim,
        )
        params = FrozenDict(net_def.init(key1, observations)["params"])
        net = TrainState.create(
            apply_fn=net_def.apply,
            params=params,
            tx=optax.adam(learning_rate=lr),
        )
        target_net = TrainState.create(
            apply_fn=net_def.apply,
            params=params,
            tx=optax.adam(learning_rate=lr),
        )

        @jax.jit
        def data_augmentation_fn(rng, observations):
            for pixel_key, depth_key in zip_longest(pixel_keys, depth_keys):
                key, rng = jax.random.split(rng)
                observations = batched_random_crop(
                    key, observations, pixel_key, frozen=True
                )
                if depth_key is not None:
                    observations = batched_random_crop(
                        key, observations, depth_key, frozen=True
                    )
            return observations

        return cls(
            rng=rng,
            net=net,
            target_net=target_net,
            data_augmentation_fn=data_augmentation_fn,
        )

    def _update(self, batch: DatasetDict) -> Tuple[struct.PyTreeNode, Dict[str, float]]:
        def loss_fn(params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            def get_v(params, s, g, z):
                phi = self.net.apply_fn({"params": params}, s)["phi"]
                psi = self.net.apply_fn({"params": params}, g)["psi"]
                T = self.net.apply_fn({"params": params}, z)["T"]
                phi_T = apply_layernorm(phi * T)
                psi_T = apply_layernorm(psi * T)
                return -1 * optax.safe_norm(phi_T - psi_T, 1e-3, axis=-1)

            V = get_v(
                params, batch["observations"], batch["goals"], batch["desired_goals"]
            )
            nV = get_v(
                self.target_net.params,
                batch["next_observations"],
                batch["goals"],
                batch["desired_goals"],
            )
            target_V = batch["rewards"] + 0.99 * batch["masks"] * nV

            V_z = get_v(
                self.target_net.params,
                batch["next_observations"],
                batch["desired_goals"],
                batch["desired_goals"],
            )
            nV_z = get_v(
                self.target_net.params,
                batch["next_observations"],
                batch["desired_goals"],
                batch["desired_goals"],
            )
            adv = batch["desired_rewards"] + 0.99 * batch["desired_masks"] * nV_z - V_z

            def expectile_fn(adv, loss, expectile):
                weight = jnp.where(adv >= 0, expectile, 1 - expectile)
                return weight * loss

            def masked_mean(x, mask):
                mask = (mask > 0).astype(jnp.float32)
                return jnp.sum(x * mask) / (1e-5 + jnp.sum(mask))

            loss = expectile_fn(adv, jnp.square(V - target_V), 0.9).mean()
            return loss, {
                "icvf_loss": loss,
                "V_success": masked_mean(V, 1.0 - batch["masks"]),
                "V_failure": masked_mean(V, batch["masks"]),
            }

        grads, info = jax.grad(loss_fn, has_aux=True)(self.net.params)
        net = self.net.apply_gradients(grads=grads)
        target_params = optax.incremental_update(
            self.net.params, self.target_net.params, 0.005
        )
        target_net = self.target_net.replace(params=target_params)
        return self.replace(net=net, target_net=target_net), info

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int):
        rng, key = jax.random.split(self.rng)
        observations = self.data_augmentation_fn(key, batch["observations"])
        rng, key = jax.random.split(rng)
        next_observations = self.data_augmentation_fn(key, batch["next_observations"])
        goals = self.data_augmentation_fn(key, batch["goals"])
        desired_goals = self.data_augmentation_fn(key, batch["desired_goals"])

        batch = batch.copy(
            add_or_replace={
                "observations": observations,
                "next_observations": next_observations,
                "goals": goals,
                "desired_goals": desired_goals,
            }
        )
        new_self = self.replace(rng=rng)

        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            new_self, info = new_self._update(mini_batch)

        return new_self, info
