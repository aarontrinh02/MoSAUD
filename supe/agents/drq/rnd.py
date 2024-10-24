from functools import partial
from typing import Callable, Dict, Optional, Sequence, Tuple

import gym
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState

from supe.agents.drq.augmentations import batched_random_crop
from supe.agents.drq.drq_learner import _unpack
from supe.data.dataset import DatasetDict
from supe.networks import (MLP, D4PGEncoder, PixelMultiplexer, StateFeature,
                           share_encoder)
from supe.types import PRNGKey


class PixelRND(struct.PyTreeNode):
    rng: PRNGKey
    net: TrainState
    frozen_net: TrainState
    coeff: float = struct.field(pytree_node=False)
    data_augmentation_fn: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        lower_agent: Optional[TrainState] = None,
        lr: float = 3e-4,
        coeff: float = 1.0,
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_filters: Sequence[int] = (3, 3, 3, 3),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        latent_dim: int = 50,
        feature_dim: int = 256,
        hidden_dims: Sequence[int] = (256, 256),
        use_icvf: bool = False,
    ):

        observations = observation_space.sample()
        action_space.sample()

        rng = jax.random.PRNGKey(seed)
        rng, key1, key2 = jax.random.split(rng, 3)

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
        rnd_cls = partial(StateFeature, base_cls=rnd_base_cls, feature_dim=feature_dim)
        net_def = PixelMultiplexer(
            pixel_encoder=pixel_encoder,
            network_cls=rnd_cls,
            stop_gradient=use_icvf,
            latent_dim=latent_dim,
        )
        params = FrozenDict(net_def.init(key1, observations)["params"])
        net = TrainState.create(
            apply_fn=net_def.apply,
            params=params,
            tx=optax.adam(learning_rate=lr),
        )
        if lower_agent is not None and not use_icvf:
            net = share_encoder(
                source=lower_agent.train_state,
                target=net,
            )

        frozen_params = FrozenDict(net_def.init(key2, observations)["params"])
        frozen_net = TrainState.create(
            apply_fn=net_def.apply,
            params=frozen_params,
            tx=optax.adam(learning_rate=lr),
        )
        if lower_agent is not None and not use_icvf:
            frozen_net = share_encoder(
                source=lower_agent.train_state,
                target=frozen_net,
            )

        @jax.jit
        def data_augmentation_fn(rng, observations):
            key, rng = jax.random.split(rng)
            observations = batched_random_crop(key, observations, "pixels")
            return observations

        return cls(
            rng=rng,
            net=net,
            frozen_net=frozen_net,
            coeff=coeff,
            data_augmentation_fn=data_augmentation_fn,
        )

    def _expand_deep(self, input: dict):
        for k, v in input.items():
            if isinstance(v, dict):
                input[k] = self._expand_deep(v)
            else:
                input[k] = v[None]
        return FrozenDict(input)

    @jax.jit
    def update(self, batch: DatasetDict) -> Tuple[struct.PyTreeNode, Dict[str, float]]:
        batch = self._expand_deep(batch)
        rng, key = jax.random.split(self.rng)
        observations = self.data_augmentation_fn(key, batch["observations"])
        rng, key = jax.random.split(rng)
        next_observations = self.data_augmentation_fn(key, batch["next_observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": observations,
                "next_observations": next_observations,
            }
        )
        new_self = self.replace(rng=rng)

        def loss_fn(params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            feats = new_self.net.apply_fn({"params": params}, batch["observations"])
            frozen_feats = new_self.frozen_net.apply_fn(
                {"params": new_self.frozen_net.params}, batch["observations"]
            )

            loss = ((feats - frozen_feats) ** 2.0).mean()
            return loss, {"rnd_loss": loss}

        grads, info = jax.grad(loss_fn, has_aux=True)(new_self.net.params)
        net = new_self.net.apply_gradients(grads=grads)

        return new_self.replace(net=net), info

    @partial(jax.jit, static_argnames="stats")
    def get_reward(self, batch, stats=False):
        if "pixels" not in batch["next_observations"]:
            batch = _unpack(batch)
        feats = self.net.apply_fn({"params": self.net.params}, batch["observations"])
        frozen_feats = self.net.apply_fn(
            {"params": self.frozen_net.params}, batch["observations"]
        )
        reward = jnp.mean((feats - frozen_feats) ** 2.0, axis=-1) * self.coeff
        if stats:
            stats = {
                "mean": reward.mean(),
                "std": reward.std(),
                "min": reward.min(),
                "max": reward.max(),
            }
            return reward, stats
        return reward
