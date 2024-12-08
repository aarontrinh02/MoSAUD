from typing import Dict, Tuple, Any
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from gym import spaces

from supe.models.common import MLP, Model, TrainState


class DynamicsNetwork(nn.Module):
    hidden_dims: Tuple[int, ...]
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, training: bool = False, rngs: Dict[str, Any] = None) -> jnp.ndarray:
        # Concatenate observation and action
        x = jnp.concatenate([obs, action], axis=-1)

        x = MLP(
            self.hidden_dims,
            activate_final=True,
            dropout_rate=self.dropout_rate,
        )(x, training=training)
        
        delta = nn.Dense(obs.shape[-1])(x)
        
        next_obs = obs + delta
        return next_obs


class DynamicsModel(Model):
    def __init__(self, network: DynamicsNetwork, optimizer: optax.GradientTransformation):
        self.network = network
        self.optimizer = optimizer

    def __call__(self, observation: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return self.network.apply(
            {"params": self.state.params},
            observation,
            action,
            training=False,
            rngs=None,
        )

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        hidden_dims: Tuple[int, ...] = (256, 256),
        learning_rate: float = 3e-4,
        weight_decay: float = 0.0,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        network = DynamicsNetwork(
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
        )

        rng = jax.random.PRNGKey(seed)
        
        dummy_obs = jnp.zeros((1,) + observation_space.shape)
        dummy_action = jnp.zeros((1,) + action_space.shape)
        params = network.init(rng, dummy_obs, dummy_action)["params"]

        optimizer = optax.adamw(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )

        state = TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=optimizer,
        )

        model = cls(network=network, optimizer=optimizer)
        model.state = state
        return model

    def update(self, batch: Dict[str, jnp.ndarray], utd_ratio: int) -> Tuple[Any, Dict[str, float]]:
        def loss_fn(params: Dict, observations: jnp.ndarray, actions: jnp.ndarray, next_observations: jnp.ndarray, rng: jnp.ndarray) -> Tuple[jnp.ndarray, Dict]:
            rngs = {'dropout': rng}
            pred_next_obs = self.network.apply(
                {"params": params},
                observations,
                actions,
                training=True,
                rngs=rngs,
            )

            loss = jnp.mean((pred_next_obs - next_observations) ** 2)

            metrics = {
                "loss": loss,
                "pred_mean": pred_next_obs.mean(),
                "pred_std": pred_next_obs.std(),
                "target_mean": next_observations.mean(),
                "target_std": next_observations.std(),
            }
            
            return loss, metrics

        observations = batch["observations"]
        actions = batch["actions"]
        next_observations = batch["next_observations"]

        rng = jax.random.PRNGKey(self.state.step)

        new_state, metrics = self._update_step(
            loss_fn,
            self.state,
            observations,
            actions,
            next_observations,
            rng,
        )

        self.state = new_state
        return self, metrics

    def get_next_observation(self, observation: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return self.network.apply(
            {"params": self.state.params},
            observation,
            action,
            training=False,
            rngs=None,
        ) 