"""
Adapted from the official codebase of IDQL -- https://github.com/philippe-eecs/IDQL
"""

from collections import OrderedDict
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState
from jax.lax import scan
from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder

from supe.networks import MLP, D4PGEncoder, MLPResNet
from supe.types import PRNGKey


def broadcast_concatenate(*arrs):
    shape = jnp.broadcast_shapes(*map(lambda x: x.shape[:-1], arrs))
    return jnp.concatenate(
        tuple(map(lambda x: jnp.broadcast_to(x, shape=shape + (x.shape[-1],)), arrs)),
        axis=-1,
    )


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = jnp.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0, 0.999)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    betas = jnp.linspace(beta_start, beta_end, timesteps)
    return betas


def vp_beta_schedule(timesteps):
    t = jnp.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.0
    b_min = 0.1
    alpha = jnp.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T**2)
    betas = 1 - alpha
    return betas


class FourierFeatures(nn.Module):
    output_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        w = self.param(
            "kernel",
            nn.initializers.normal(0.2),
            (self.output_size // 2, 1),
            jnp.float32,
        )
        f = 2 * jnp.pi * x @ w.T
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)


class DDPM(nn.Module):
    cond_encoder_cls: Type[nn.Module]
    reverse_encoder_cls: Type[nn.Module]
    time_preprocess_cls: Type[nn.Module]
    pixel_encoder: D4PGEncoder = None
    latent_dim: int = 50
    default_init: Callable = nn.initializers.xavier_uniform

    @nn.compact
    def __call__(
        self, s: jnp.ndarray, a: jnp.ndarray, time: jnp.ndarray, training: bool = False
    ):

        if self.pixel_encoder is not None:
            obs_pixels = s["pixels"]
            obs_pixels = obs_pixels.astype(jnp.float32) / 255.0
            obs_pixels = jnp.reshape(
                obs_pixels, (*obs_pixels.shape[:-2], -1)
            )  # remove stacking dimension
            s = self.pixel_encoder(obs_pixels)
            s = nn.Dense(self.latent_dim, kernel_init=self.default_init())(s)
            s = nn.LayerNorm()(s)
            s = nn.tanh(s)
        t_ff = self.time_preprocess_cls()(time)
        cond = self.cond_encoder_cls()(t_ff, training=training)
        reverse_input = broadcast_concatenate(a, s, cond)

        return self.reverse_encoder_cls()(reverse_input, training=training)


class DiffusionBC(struct.PyTreeNode):
    rng: jax.random.PRNGKey
    train_state: TrainState
    target_params: Any
    tau: float = struct.field(pytree_node=False)
    betas: Any = struct.field(pytree_node=False)
    alpha_hats: Any = struct.field(pytree_node=False)
    alphas: Any = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()

        config.beta_schedule = "vp"
        config.T = 5
        config.use_layer_norm = True
        config.num_blocks = 3
        config.dropout_rate = 0.1
        config.tau = 0.001
        config.hidden_dim = 128
        config.lr = 3e-4
        config.decay_steps = 3e6

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def net(self, params, *args, **kwargs):
        return self.train_state.apply_fn({"params": params}, *args, **kwargs)

    @classmethod
    def create(cls, config, rng, observations, actions):
        action_dim = actions.shape[-1]
        preprocess_time_cls = partial(FourierFeatures, output_size=config.hidden_dim)

        if type(observations) == OrderedDict:
            pixel_encoder = D4PGEncoder(
                features=config.cnn_features,
                filters=config.cnn_filters,
                strides=config.cnn_strides,
                padding=config.cnn_padding,
            )
        else:
            pixel_encoder = None

        cond_model_cls = partial(
            MLP,
            hidden_dims=(config.hidden_dim * 2, config.hidden_dim * 2),
            activations=nn.swish,
            activate_final=False,
        )

        base_model_cls = partial(
            MLPResNet,
            use_layer_norm=config.use_layer_norm,
            num_blocks=config.num_blocks,
            dropout_rate=config.dropout_rate,
            out_dim=action_dim,
            activations=nn.swish,
        )

        model_def = DDPM(
            pixel_encoder=pixel_encoder,
            time_preprocess_cls=preprocess_time_cls,
            cond_encoder_cls=cond_model_cls,
            reverse_encoder_cls=base_model_cls,
        )

        model_key, rng = jax.random.split(rng)
        params = model_def.init(model_key, observations, actions, jnp.zeros((1, 1)))[
            "params"
        ]

        if config.decay_steps is not None:
            lr = optax.cosine_decay_schedule(config.lr, config.decay_steps)
        else:
            lr = config.lr

        train_state = TrainState.create(
            apply_fn=model_def.apply, params=params, tx=optax.adamw(learning_rate=lr)
        )

        if config.beta_schedule == "cosine":
            betas = jnp.array(cosine_beta_schedule(config.T))
        elif config.beta_schedule == "linear":
            betas = jnp.linspace(1e-4, 2e-2, config.T)
        elif config.beta_schedule == "vp":
            betas = jnp.array(vp_beta_schedule(config.T))
        else:
            raise ValueError(f"Invalid beta schedule: {config.beta_schedule}")

        alphas = 1 - betas
        alpha_hat = jnp.array([jnp.prod(alphas[: i + 1]) for i in range(config.T)])

        return cls(
            rng=rng,
            train_state=train_state,
            target_params=deepcopy(params),
            betas=betas,
            alpha_hats=alpha_hat,
            alphas=alphas,
            T=config.T,
            action_dim=action_dim,
            tau=config.tau,
        )

    @staticmethod
    def _update(self, batch):
        batch_shape = batch["actions"].shape[:-1]
        rng = self.rng
        time_key, noise_key, dropout_key, rng = jax.random.split(rng, 4)
        time = jax.random.randint(time_key, batch_shape, 0, self.T)
        noise_sample = jax.random.normal(noise_key, batch["actions"].shape)

        alpha_hats = self.alpha_hats[time]
        time = jnp.expand_dims(time, axis=1)
        alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
        alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
        noisy_actions = alpha_1 * batch["actions"] + alpha_2 * noise_sample

        def loss_fn(params):
            eps_pred = self.net(
                params,
                batch["observations"],
                noisy_actions,
                time,
                True,
                rngs={"dropout": dropout_key},
            )

            loss = (((eps_pred - noise_sample) ** 2).sum(axis=-1)).mean()
            return loss, {"ddpm_loss": loss}

        grads, info = jax.grad(loss_fn, has_aux=True)(self.train_state.params)
        new_train_state = self.train_state.apply_gradients(grads=grads)

        target_params = optax.incremental_update(
            new_train_state.params, self.target_params, self.tau
        )
        return (
            self.replace(
                train_state=new_train_state, target_params=target_params, rng=rng
            ),
            info,
        )

    @partial(jax.jit, static_argnames=("utd_ratio", "aux"))
    def update(self, batch, utd_ratio: int = 1, aux: bool = False):

        batch = jax.tree_util.tree_map(
            lambda x: x.reshape(utd_ratio, x.shape[0] // utd_ratio, *x.shape[1:]), batch
        )
        new_agent, infos = scan(self._update, self, batch)
        if aux:
            info = jax.tree_util.tree_map(lambda x: x.mean(), infos)
            return new_agent, info
        return new_agent, {}

    @partial(
        jax.jit,
        static_argnames=(
            "temperature",
            "repeat_last_step",
            "clip_sampler",
            "sample_shape",
        ),
    )
    def sample_actions(
        self,
        rng,
        observations,
        sample_shape=(),
        temperature=1.0,
        repeat_last_step=0,
        clip_sampler=True,
    ):
        if type(observations) == dict:
            batch_shape = observations["pixels"].shape[:-4]
        else:
            batch_shape = observations.shape[:-1]
        key1, key2 = jax.random.split(rng)

        def fn(input_tuple, time):
            current_x, rng = input_tuple

            eps_pred = self.net(
                self.target_params, observations, current_x, time[None], False
            )

            alpha_1 = 1 / jnp.sqrt(self.alphas[time])
            alpha_2 = (1 - self.alphas[time]) / (jnp.sqrt(1 - self.alpha_hats[time]))
            current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

            rng, key = jax.random.split(rng, 2)
            z = jax.random.normal(key, shape=current_x.shape)
            z_scaled = temperature * z
            current_x = current_x + (time > 0) * (jnp.sqrt(self.betas[time]) * z_scaled)

            if clip_sampler:
                current_x = jnp.clip(current_x, -1, 1)

            return (current_x, rng), ()

        input_tuple, () = jax.lax.scan(
            fn,
            (
                jax.random.normal(
                    key1, sample_shape + batch_shape + (self.action_dim,)
                ),
                key2,
            ),
            jnp.arange(self.T - 1, -1, -1),
        )

        for _ in range(repeat_last_step):
            input_tuple, () = fn(input_tuple, 0)

        action_0, rng = input_tuple
        action_0 = jnp.clip(action_0, -1, 1)

        return action_0
