from copy import deepcopy
from functools import partial
from typing import Any, Dict, Optional

import jax
import jax.nn as nn
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState
from jax.lax import scan

from supe.distributions import Normal
from supe.networks import MLP, Ensemble, StateActionValue, StateValue


def default_init(scale: Optional[float] = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


MLP = partial(MLP, default_init=default_init)
StateActionValue = partial(StateActionValue, default_init=default_init)
StateValue = partial(StateValue, default_init=default_init)
Normal = partial(Normal, default_init=default_init)


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


class IQL(struct.PyTreeNode):
    rng: jax.random.PRNGKey
    train_states: Dict[str, TrainState]
    target_params: Any
    tau: float = struct.field(pytree_node=False)
    discount: float = struct.field(pytree_node=False)
    expectile: float = struct.field(pytree_node=False)
    temperature: float = struct.field(pytree_node=False)
    num_qs: int = struct.field(pytree_node=False)

    observation_dim: int = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)

    policy_extraction: str = struct.field(pytree_node=False)
    faster_actor_update: bool = struct.field(pytree_node=False)

    def actor(self, params, *args, **kwargs):
        return self.train_states["actor"].apply_fn({"params": params}, *args, **kwargs)

    def critic(self, params, *args, **kwargs):
        return self.train_states["critic"].apply_fn({"params": params}, *args, **kwargs)

    def value(self, params, *args, **kwargs):
        return self.train_states["value"].apply_fn({"params": params}, *args, **kwargs)

    @property
    def train_state_keys(self):
        return list(self.train_states)

    @property
    def n_train_states(self):
        return len(self.train_state_keys)

    @staticmethod
    def initialize(config, rng, observation_dim, action_dim):
        observations, actions = jnp.zeros((10, observation_dim)), jnp.zeros(
            (10, action_dim)
        )
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)
        actor_opt = optax.adam(learning_rate=config.lr)
        actor_base_cls = partial(
            MLP,
            hidden_dims=config.hidden_dims,
            activate_final=True,
            scale_final=1e-2,
        )
        actor_def = Normal(
            actor_base_cls,
            action_dim,
            log_std_min=-5.0,
            state_dependent_std=False,
            squash_tanh=False,
        )
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply, params=actor_params, tx=actor_opt
        )

        critic_base_cls = partial(
            MLP,
            hidden_dims=config.hidden_dims,
            activate_final=True,
            use_layer_norm=config.critic_layer_norm,
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=config.num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=config.lr),
        )

        value_base_cls = partial(
            MLP,
            hidden_dims=config.hidden_dims,
            activate_final=True,
            use_layer_norm=config.critic_layer_norm,
        )
        value_def = StateValue(base_cls=value_base_cls)
        value_params = value_def.init(value_key, observations)["params"]
        value = TrainState.create(
            apply_fn=value_def.apply,
            params=value_params,
            tx=optax.adam(learning_rate=config.lr),
        )

        train_states = {"critic": critic, "value": value, "actor": actor}
        target_params = {
            key: deepcopy(train_states[key].params) for key in train_states
        }

        return train_states, target_params

    @classmethod
    def create(cls, config, rng, observation_dim, action_dim):
        train_states, target_params = cls.initialize(
            config, rng, observation_dim, action_dim
        )

        return cls(
            rng=rng,
            observation_dim=observation_dim,
            action_dim=action_dim,
            train_states=train_states,
            target_params=target_params,
            tau=config.tau,
            discount=config.discount,
            num_qs=config.num_qs,
            expectile=config.expectile,
            temperature=config.temperature,
            policy_extraction=config.policy_extraction,
            faster_actor_update=config.faster_actor_update,
        )

    def reset(self, config):
        init_rng, rng = jax.random.split(self.rng)
        new_train_states, new_target_params = IQL.initialize(
            config, init_rng, self.observation_dim, self.action_dim
        )
        return self.replace(
            rng=rng, train_states=new_train_states, target_params=new_target_params
        )

    @jax.jit
    def eval_actions(self, observations):
        dist = self.actor(self.train_states["actor"].params, observations)
        return jnp.clip(dist.mode(), -1.0, 1.0)

    @jax.jit
    def sample_actions(self, rng, observations):
        dist = self.actor(self.train_states["actor"].params, observations)
        return jnp.clip(dist.sample(seed=rng), -1.0, 1.0)

    @staticmethod
    def _update(self, batch):

        def loss_fn(params):

            a_dist = self.actor(params["actor"], batch["observations"])
            value = self.value(params["value"], batch["observations"])

            # faster actor update from Seohong to not use target critic
            target_q_values = jnp.min(
                self.critic(
                    self.target_params["critic"],
                    batch["observations"],
                    batch["actions"],
                ),
                axis=0,
            )
            target_adv = target_q_values - value

            q_values = jnp.min(
                self.critic(params["critic"], batch["observations"], batch["actions"]),
                axis=0,
            )
            adv = q_values - value

            """ Policy loss """
            exp_a = jnp.exp(
                (adv if self.faster_actor_update else target_adv) * self.temperature
            )
            exp_a = jnp.minimum(exp_a, 100.0)

            b_log_pi = a_dist.log_prob(batch["actions"])

            """ V function loss """
            weight = jnp.where(target_adv > 0, self.expectile, (1.0 - self.expectile))
            value_loss = (weight * (target_adv**2.0)).mean()

            """ Q function loss """
            next_value = self.value(params["value"], batch["next_observations"])
            qs = self.critic(params["critic"], batch["observations"], batch["actions"])

            q_target = batch["rewards"] + batch["masks"] * self.discount * next_value
            critic_loss = jnp.mean(jnp.sum((qs - q_target) ** 2.0, axis=0))

            if self.policy_extraction == "awr":
                actor_loss = -(exp_a * b_log_pi).mean()
            elif self.policy_extraction == "ddpg":
                actor_loss = -qs.mean()

            losses = {"actor": actor_loss, "critic": critic_loss, "value": value_loss}

            info = {
                "actor_loss": actor_loss,
                "v": value.mean(),
                "value_loss": value_loss,
                "adv": target_adv.mean(),
                "critic_loss": critic_loss,
                "q": qs.mean(),
            }
            losses = {"actor": actor_loss, "critic": critic_loss, "value": value_loss}
            return tuple(losses[key] for key in self.train_state_keys), info

        # compute gradient
        train_params = {
            key: self.train_states[key].params for key in self.train_state_keys
        }
        (_, info), grads = value_and_multi_grad(loss_fn, self.n_train_states)(
            train_params
        )

        # apply gradient and target param update
        new_train_states = {
            key: self.train_states[key].apply_gradients(grads=grad[key])
            for key, grad in zip(self.train_state_keys, grads)
        }
        new_target_params = {
            key: update_target_network(
                new_train_states[key].params, self.target_params[key], self.tau
            )
            for key in self.train_state_keys
        }

        return (
            self.replace(
                train_states=new_train_states, target_params=new_target_params
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
