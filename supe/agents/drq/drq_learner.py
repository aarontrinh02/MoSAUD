"""Implementations of algorithms for continuous control."""

from functools import partial
from itertools import zip_longest
from typing import Callable, Optional, Sequence, Tuple

import gym
import jax
import optax
from flax import struct
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from supe.agents.drq.augmentations import batched_random_crop
from supe.agents.sac.sac_learner import SACLearner
from supe.agents.sac.temperature import Temperature
from supe.data.dataset import DatasetDict
from supe.distributions import TanhNormal
from supe.networks import MLP, Ensemble, PixelMultiplexer, StateActionValue
from supe.networks.encoders import D4PGEncoder
from supe.networks.pixel_multiplexer import share_encoder


# Helps to minimize CPU to GPU transfer.
def _unpack(batch):
    # Assuming that if next_observation is missing, it's combined with observation:
    for pixel_key in batch["observations"].keys():
        if pixel_key not in batch["next_observations"]:
            obs_pixels = batch["observations"][pixel_key][..., :-1]
            next_obs_pixels = batch["observations"][pixel_key][..., 1:]

            obs = batch["observations"].copy(add_or_replace={pixel_key: obs_pixels})
            next_obs = batch["next_observations"].copy(
                add_or_replace={pixel_key: next_obs_pixels}
            )

    batch = batch.copy(
        add_or_replace={"observations": obs, "next_observations": next_obs}
    )

    return batch


class DrQLearner(SACLearner):
    data_augmentation_fn: Callable = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        lower_agent: Optional[TrainState] = None,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        cnn_features: Sequence[int] = (32, 32, 32, 32),
        cnn_filters: Sequence[int] = (3, 3, 3, 3),
        cnn_strides: Sequence[int] = (2, 1, 1, 1),
        cnn_padding: str = "VALID",
        latent_dim: int = 50,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
        pixel_keys: Tuple[str, ...] = ("pixels",),
        depth_keys: Tuple[str, ...] = (),
        weight_decay: float = 1e-3,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        if target_entropy is None:
            target_entropy = -action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        pixel_encoder = D4PGEncoder(
            features=cnn_features,
            filters=cnn_filters,
            strides=cnn_strides,
            padding=cnn_padding,
        )

        actor_base_cls = partial(MLP, hidden_dims=hidden_dims, activate_final=True)
        actor_cls = partial(TanhNormal, base_cls=actor_base_cls, action_dim=action_dim)
        actor_def = PixelMultiplexer(
            pixel_encoder=pixel_encoder,
            network_cls=actor_cls,
            stop_gradient=True,
            latent_dim=latent_dim,
        )
        actor_params = FrozenDict(actor_def.init(actor_key, observations)["params"])
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adamw(learning_rate=actor_lr, weight_decay=weight_decay),
        )

        if lower_agent is not None:
            actor = share_encoder(
                source=lower_agent.train_state,
                target=actor,
            )

        critic_base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
            use_layer_norm=critic_layer_norm,
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_cls = partial(Ensemble, net_cls=critic_cls, num=num_qs)
        critic_def = PixelMultiplexer(
            pixel_encoder=pixel_encoder,
            network_cls=critic_cls,
            stop_gradient=False,
            latent_dim=latent_dim,
        )
        critic_params = FrozenDict(
            critic_def.init(critic_key, observations, actions)["params"]
        )
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adamw(learning_rate=critic_lr, weight_decay=weight_decay),
        )

        if lower_agent is not None:
            critic = share_encoder(
                source=lower_agent.train_state,
                target=critic,
            )

        target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        temp_def = Temperature(init_temperature)
        temp_params = FrozenDict(temp_def.init(temp_key)["params"])
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
        )

        def data_augmentation_fn(rng, observations):
            for pixel_key, depth_key in zip_longest(pixel_keys, depth_keys):
                key, rng = jax.random.split(rng)
                observations = batched_random_crop(key, observations, pixel_key)
                if depth_key is not None:
                    observations = batched_random_crop(key, observations, depth_key)
            return observations

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            target_entropy=target_entropy,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
            data_augmentation_fn=data_augmentation_fn,
        )

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int):
        new_agent = self

        if "pixels" not in batch["next_observations"]:
            batch = _unpack(batch)

        actor = share_encoder(
            source=new_agent.critic,
            target=new_agent.actor,
        )
        new_agent = new_agent.replace(actor=actor)  # use critic CNN encoders for actor

        rng, key = jax.random.split(new_agent.rng)
        observations = self.data_augmentation_fn(key, batch["observations"])
        rng, key = jax.random.split(rng)
        next_observations = self.data_augmentation_fn(key, batch["next_observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": observations,
                "next_observations": next_observations,
            }
        )

        new_agent = new_agent.replace(rng=rng)

        return SACLearner.update(new_agent, batch, utd_ratio)
