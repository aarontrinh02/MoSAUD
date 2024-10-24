"""
Modified from Seohong's OPAL implementation in offline METRA
"""

from functools import partial
from typing import Optional, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState

from supe.networks import MLP
from supe.networks.encoders.d4pg_encoder import D4PGEncoder
from supe.utils import get_observation_at_index_in_chunk

from .iql import IQL


def default_init(scale: Optional[float] = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


MLP = partial(MLP, default_init=default_init)  # default init is xavier uniform


class SimpleGRU(nn.Module):
    hidden_size: int

    def setup(self):
        self.gru = nn.GRUCell(features=self.hidden_size)

    @partial(
        nn.transforms.scan,
        variable_broadcast="params",
        in_axes=1,
        out_axes=1,
        split_rngs={"params": False},
    )
    def __call__(self, carry, x):
        return self.gru(carry, x)


class SimpleBiGRU(nn.Module):
    hidden_size: int

    def setup(self):
        self.forward_gru = SimpleGRU(self.hidden_size)
        self.backward_gru = SimpleGRU(self.hidden_size)

    def __call__(self, embedded_inputs):
        shape = embedded_inputs[:, 0].shape

        initial_state = self.forward_gru.gru.initialize_carry(jax.random.key(0), shape)
        _, forward_outputs = self.forward_gru(initial_state, embedded_inputs)

        reversed_inputs = embedded_inputs[:, ::-1, :]
        initial_state = self.backward_gru.gru.initialize_carry(jax.random.key(0), shape)
        _, backward_outputs = self.backward_gru(initial_state, reversed_inputs)
        backward_outputs = backward_outputs[:, ::-1, :]

        outputs = jnp.concatenate([forward_outputs, backward_outputs], -1)
        return outputs


class SeqEncoder(nn.Module):
    num_recur_layers: int = 2
    output_dim: int = 2
    recur_output: str = "concat"
    hidden_size: int = 256

    def setup(self) -> None:
        self.obs_mlp = MLP([self.hidden_size, self.hidden_size], activate_final=True)
        self.recurs = [
            SimpleBiGRU(self.hidden_size) for _ in range(self.num_recur_layers)
        ]
        self.projection = MLP([self.output_dim], activate_final=False)

    def __call__(
        self,
        seq_observations: jnp.ndarray,
        seq_actions: jnp.ndarray,
    ):
        B, C, D = seq_observations.shape
        observations = jnp.reshape(seq_observations, (B * C, D))
        outputs = jnp.reshape(self.obs_mlp(observations), (B, C, -1))
        outputs = jnp.concatenate([outputs, seq_actions], axis=-1)

        for recur in self.recurs:
            outputs = recur(outputs)
        if self.recur_output == "concat":
            outputs = jnp.reshape(outputs, (B, -1))
        else:
            outputs = outputs[:, -1]
        outputs = self.projection(outputs)

        return outputs


class GaussianModule(nn.Module):
    hidden_dims: Sequence[int]
    output_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    final_fc_init_scale: float = 1e-2

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        temperature: float = 1.0,
    ) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims, activate_final=True)(inputs)

        means = nn.Dense(
            self.output_dim, kernel_init=default_init(self.final_fc_init_scale)
        )(outputs)
        log_stds = nn.Dense(
            self.output_dim, kernel_init=default_init(self.final_fc_init_scale)
        )(outputs)

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )

        return distribution


class VAE(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    skill_dim: int
    recur_output: str
    hidden_size: int
    cnn: bool
    latent_dim: int
    cnn_features: Sequence[int] = (32, 64, 128, 256)
    cnn_filters: Sequence[int] = (3, 3, 3, 3)
    cnn_strides: Sequence[int] = (2, 2, 2, 2)
    cnn_padding: str = "VALID"

    def setup(self) -> None:
        cnn_features: Sequence[int] = (32, 64, 128, 256)
        cnn_filters: Sequence[int] = (3, 3, 3, 3)
        cnn_strides: Sequence[int] = (2, 2, 2, 2)
        cnn_padding: str = "VALID"
        if self.cnn:
            self.pixel_encoder = D4PGEncoder(
                features=cnn_features,
                filters=cnn_filters,
                strides=cnn_strides,
                padding=cnn_padding,
            )
            self.fc_image = nn.Dense(self.latent_dim, kernel_init=default_init())
            self.fc_state = nn.Dense(self.latent_dim, kernel_init=default_init())
            self.layer_norm_image = nn.LayerNorm()
            self.layer_norm_state = nn.LayerNorm()

        self.seq_encoder = SeqEncoder(
            num_recur_layers=2,
            output_dim=self.skill_dim * 2,
            recur_output=self.recur_output,
            hidden_size=self.hidden_size,
        )

        self.prior_model = GaussianModule(self.hidden_dims, self.skill_dim)

        self.recon_model = GaussianModule(self.hidden_dims, self.action_dim)

    def encode_obs(self, observations, stop_gradient=False):
        if self.cnn:
            obs_pixels = observations["pixels"]
            length = len(obs_pixels.shape)
            if length == 6:  # B, Chunk, H, W, Channel, 1
                B, C, *obs_shape = obs_pixels.shape  # NOQA
                obs_pixels = obs_pixels.reshape((B * C, *obs_pixels.shape[2:]))

            obs_pixels = obs_pixels.astype(jnp.float32) / 255.0

            if obs_pixels.shape[-1] == 1:
                obs_pixels = jnp.reshape(
                    obs_pixels, (*obs_pixels.shape[:-2], -1)
                )  # Remove stacking dimension, if present

            obs_pixels = self.pixel_encoder(obs_pixels)

            if stop_gradient:
                obs_pixels = jax.lax.stop_gradient(obs_pixels)

            if length == 6:
                obs_pixels = obs_pixels.reshape((B, C, *obs_pixels.shape[1:]))

            obs_pixels = self.fc_image(obs_pixels)
            obs_pixels = self.layer_norm_image(obs_pixels)
            obs_pixels = nn.tanh(obs_pixels)

            if "state" in observations:
                obs_state = observations["state"]
                obs_state = self.fc_state(obs_state)
                obs_state = self.layer_norm_state(obs_state)
                obs_state = nn.tanh(obs_state)

                observations = jnp.concatenate([obs_pixels, obs_state], axis=-1)
            else:
                observations = obs_pixels

        return observations

    def encode(self, seq_observations: jnp.ndarray, seq_actions: jnp.ndarray):
        seq_observations = self.encode_obs(observations=seq_observations)
        outputs = self.seq_encoder(seq_observations, seq_actions)
        return outputs[..., : self.skill_dim]

    def act(
        self,
        observations: jnp.ndarray,
        skills: jnp.array,
        temperature: float = 1.0,
    ) -> distrax.Distribution:
        observations = self.encode_obs(observations)

        szs = jnp.concatenate([observations, skills], axis=-1)
        action_dists = self.recon_model(szs, temperature=temperature)

        return action_dists

    def prior(self, observations: jnp.ndarray):
        observations = self.encode_obs(observations)
        return self.prior_model(observations)

    def __call__(
        self,
        seq_observations: jnp.ndarray,
        seq_actions: jnp.ndarray,
        z_rng,
    ):
        seq_observations = self.encode_obs(observations=seq_observations)
        B, C, D = seq_observations.shape
        outputs = self.seq_encoder(seq_observations, seq_actions)
        means = outputs[..., : self.skill_dim]
        log_stds = outputs[..., self.skill_dim :]
        stds = jnp.exp(0.5 * log_stds)
        posteriors = distrax.MultivariateNormalDiag(loc=means, scale_diag=stds)

        priors = self.prior_model(
            seq_observations[:, 0]
        )  # Batch Size x Chunk Size x Observation Dim

        zs = means + stds * jax.random.normal(z_rng, means.shape)
        zs = jnp.expand_dims(zs, axis=1).repeat(C, axis=1)
        szs = jnp.concatenate([seq_observations, zs], axis=-1)
        recon_action_dists = self.recon_model(szs)

        return recon_action_dists, priors, posteriors


class OPAL(flax.struct.PyTreeNode):
    rng: jax.random.PRNGKey
    train_state: TrainState
    iql: flax.struct.PyTreeNode
    chunk_size: int = flax.struct.field(pytree_node=False)
    kl_coef: float = flax.struct.field(pytree_node=False)
    beta_coef: float = flax.struct.field(pytree_node=False)
    discount: float = flax.struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        config,
        rng,
        observations,
        actions,
        chunk_size,
        cnn=False,
    ):
        rng, vae_key, iql_key = jax.random.split(rng, 3)

        if cnn:
            seq_observations = {}
            for key in observations:
                if len(observations[key].shape) == 4:
                    seq_observations[key] = np.tile(
                        observations[key], (1, chunk_size, 1, 1, 1, 1)
                    )
                elif len(observations[key].shape) == 1:
                    seq_observations[key] = np.tile(
                        observations[key], (1, chunk_size, 1)
                    )
                else:
                    raise ValueError("Invalid observation shape")
        else:
            seq_observations = np.tile(observations, (1, chunk_size, 1))

        seq_actions = np.tile(actions, (1, chunk_size, 1))

        vae_tx = optax.adam(learning_rate=config.lr)
        vae_def = VAE(
            hidden_dims=config.vae_hidden_dims,
            action_dim=seq_actions.shape[-1],
            skill_dim=config.skill_dim,
            recur_output="concat",
            hidden_size=config.vae_encoder_hidden_size,
            cnn=cnn,
            latent_dim=config.latent_dim,
        )

        vae_key, z_key = jax.random.split(vae_key)
        vae_params = vae_def.init(vae_key, seq_observations, seq_actions, z_key)[
            "params"
        ]

        train_state = TrainState.create(
            apply_fn=vae_def.apply, params=vae_params, tx=vae_tx
        )

        skills = np.zeros((1, config.skill_dim))

        if cnn:
            observation_dim = 2 * config.latent_dim
        else:
            observation_dim = observations.shape[-1]

        iql = IQL.create(config.iql, iql_key, observation_dim, skills.shape[-1])

        return cls(
            rng=rng,
            train_state=train_state,
            iql=iql,
            chunk_size=chunk_size,
            kl_coef=config.kl_coef,
            beta_coef=config.beta_coef,
            discount=config.discount,
        )

    def vae(self, params, *args, **kwargs):
        return self.train_state.apply_fn({"params": params}, *args, **kwargs)

    @jax.jit
    def prior_model(self, observations):
        return self.vae(self.train_state.params, observations, method="prior")

    @partial(jax.jit, static_argnames=("aux",))
    def update_vae(agent, batch, aux=False):
        rng, z_key = jax.random.split(agent.rng)

        def vae_loss_fn(vae_params):
            recon_action_dists, priors, posteriors = agent.vae(
                vae_params, batch["seq_observations"], batch["seq_actions"], z_key
            )
            recon_loss = -recon_action_dists.log_prob(batch["seq_actions"]).mean()
            kl_loss = posteriors.kl_divergence(priors).mean()
            total_loss = recon_loss + agent.kl_coef * kl_loss
            return total_loss, {
                "recon_loss": recon_loss,
                "kl_loss": kl_loss,
                "total_loss": total_loss,
                "prior_mean": priors.loc.mean(),
                "prior_std": priors.scale_diag.mean(),
                "posterior_mean": posteriors.loc.mean(),
                "posterior_std": posteriors.scale_diag.mean(),
            }

        (loss, info), grads = jax.value_and_grad(vae_loss_fn, has_aux=True)(
            agent.train_state.params
        )
        new_train_state = agent.train_state.apply_gradients(grads=grads)

        return agent.replace(train_state=new_train_state, rng=rng), info if aux else {}

    @partial(jax.jit, static_argnames=("aux",))
    def update_iql(agent, batch, aux=False):
        skills = agent.vae(
            agent.train_state.params,
            batch["seq_observations"],
            batch["seq_actions"],
            method="encode",
        )

        reward_discounts = jnp.power(agent.discount, jnp.arange(agent.chunk_size))
        seq_rewards = batch["seq_rewards"] * reward_discounts
        seq_rewards = jnp.concatenate(
            [seq_rewards[:, [0]], seq_rewards[:, 1:] * batch["seq_masks"][:, :-1]],
            axis=-1,
        )
        rewards = seq_rewards.sum(axis=-1)

        masks = jnp.min(batch["seq_masks"], axis=1)

        observations = get_observation_at_index_in_chunk(
            batch["seq_observations"], index=0
        )
        next_observations = get_observation_at_index_in_chunk(
            batch["next_seq_observations"], index=-1
        )

        if type(observations) == dict:
            next_observations = agent.vae(
                agent.train_state.params, next_observations, method="encode_obs"
            )
            observations = agent.vae(
                agent.train_state.params, observations, method="encode_obs"
            )

        iql_batch = FrozenDict(
            {
                "actions": skills,
                "next_observations": next_observations,
                "observations": observations,
                "rewards": rewards,
                "masks": masks,
            }
        )

        new_iql, iql_info = agent.iql.update(iql_batch, aux=aux)
        return agent.replace(iql=new_iql), iql_info

    @jax.jit
    def sample_skills(agent, rng, observations):
        observations = agent.vae(
            agent.train_state.params, observations, method="encode_obs"
        )
        observations = jax.lax.stop_gradient(
            observations
        )  # we don't want to change the encoder with IQL

        dist = agent.iql.actor(agent.iql.train_states["actor"].params, observations)
        return dist.sample(seed=rng)

    @jax.jit
    def eval_skills(agent, observations):
        observations = agent.vae(
            agent.train_state.params, observations, method="encode_obs"
        )
        observations = jax.lax.stop_gradient(observations)

        dist = agent.iql.actor(agent.iql.train_states["actor"].params, observations)
        return dist.mode()

    @jax.jit
    def sample_skill_actions(agent, rng, observations, skills):
        actions = agent.vae(
            agent.train_state.params, observations, skills=skills, method="act"
        ).sample(seed=rng)
        actions = jnp.clip(actions, -1.0, 1.0)
        return actions

    @jax.jit
    def eval_skill_actions(agent, observations, skills):
        actions = agent.vae(
            agent.train_state.params, observations, skills=skills, method="act"
        ).mode()
        actions = jnp.clip(actions, -1.0, 1.0)
        return actions

    @jax.jit
    def sample_actions(agent, rng, observations):
        rng_skill, rng_action = jax.random.split(rng)
        skills = agent.sample_skills(rng_skill, observations)
        actions = agent.sample_skill_actions(rng_action, observations, skills)
        return actions

    @jax.jit
    def eval_actions(agent, observations):
        skills = agent.iql.eval_actions(observations)
        actions = agent.vae(
            agent.train_state.params, observations, skills=skills, method="act"
        ).mode()
        return actions
