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
from supe.networks import MLP, PixelMultiplexer, StateValue, share_encoder
from supe.networks.encoders import D4PGEncoder
from supe.types import PRNGKey


class PixelRM(struct.PyTreeNode):
    rng: PRNGKey
    r_net: TrainState
    m_net: TrainState
    data_augmentation_fn: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        lower_agent: Optional[TrainState] = None,
        lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_filters: Sequence[int] = (3, 3, 3, 3),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        latent_dim: int = 50,
        use_icvf: bool = False,
    ):

        observations = observation_space.sample()
        action_space.sample()

        rng = jax.random.PRNGKey(seed)
        rng, key = jax.random.split(rng)

        pixel_encoder = D4PGEncoder(
            features=cnn_features,
            filters=cnn_filters,
            strides=cnn_strides,
            padding=cnn_padding,
        )

        base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
        )
        net_cls = partial(StateValue, base_cls=base_cls)
        ucb_def = PixelMultiplexer(
            pixel_encoder=pixel_encoder,
            network_cls=net_cls,
            stop_gradient=use_icvf,
            latent_dim=latent_dim,
        )
        r_params = FrozenDict(ucb_def.init(key, observations)["params"])
        r_net = TrainState.create(
            apply_fn=ucb_def.apply,
            params=r_params,
            tx=optax.adam(learning_rate=lr),
        )
        if lower_agent is not None and not use_icvf:
            r_net = share_encoder(
                source=lower_agent.train_state,
                target=r_net,
            )

        m_params = FrozenDict(ucb_def.init(key, observations)["params"])
        m_net = TrainState.create(
            apply_fn=ucb_def.apply,
            params=m_params,
            tx=optax.adam(learning_rate=lr),
        )
        if lower_agent is not None and not use_icvf:
            m_net = share_encoder(
                source=lower_agent.train_state,
                target=m_net,
            )

        @jax.jit
        def data_augmentation_fn(rng, observations):
            key, rng = jax.random.split(rng)
            observations = batched_random_crop(key, observations, "pixels")
            return observations

        return cls(
            rng=rng,
            r_net=r_net,
            m_net=m_net,
            data_augmentation_fn=data_augmentation_fn,
        )

    def _update(self, batch: DatasetDict) -> Tuple[struct.PyTreeNode, Dict[str, float]]:
        def r_loss_fn(r_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            rs = self.r_net.apply_fn({"params": r_params}, batch["observations"])

            loss = ((rs - batch["rewards"]) ** 2.0).mean()
            return loss, {"r_loss": loss}

        grads, r_info = jax.grad(r_loss_fn, has_aux=True)(self.r_net.params)
        r_net = self.r_net.apply_gradients(grads=grads)

        def m_loss_fn(m_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            ms = self.m_net.apply_fn({"params": m_params}, batch["observations"])

            loss = optax.sigmoid_binary_cross_entropy(ms, batch["masks"]).mean()
            return loss, {"m_loss": loss}

        grads, m_info = jax.grad(m_loss_fn, has_aux=True)(self.m_net.params)
        m_net = self.m_net.apply_gradients(grads=grads)

        return self.replace(r_net=r_net, m_net=m_net), {**r_info, **m_info}

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int):

        if "pixels" not in batch["next_observations"]:
            batch = _unpack(batch)

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

        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            new_self, info = new_self._update(mini_batch)

        return new_self, info

    @jax.jit
    def get_reward(self, batch):
        if "pixels" not in batch["next_observations"]:
            batch = _unpack(batch)

        rewards = self.r_net.apply_fn(
            {"params": self.r_net.params}, batch["observations"]
        )
        return rewards

    @jax.jit
    def get_mask(self, batch):
        if "pixels" not in batch["next_observations"]:
            batch = _unpack(batch)

        logits = self.m_net.apply_fn(
            {"params": self.m_net.params}, batch["observations"]
        )
        return jax.nn.sigmoid(logits)
