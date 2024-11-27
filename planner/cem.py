from functools import partial
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from typing import Tuple, Dict, Any

class CEMPlanner(struct.PyTreeNode):
    num_policy_traj: int = 64
    num_sample_traj: int = 512
    num_elites: int = 64
    cem_iter: int = 5
    cem_temperature: float = 10.0
    cem_momentum: float = 0.1
    horizon: int = 5
    discount: float = 0.99
    min_std: float = 0.1
    action_dim: int = None
    
    @partial(jax.jit, static_argnames=('self', 'sac_agent', 'dynamics_model', 'rm'))
    def estimate_value(self, sac_agent, dynamics_model, rm, state: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """Estimate value of a trajectory using dynamics model, reward module, and SAC critic."""
        def step_fn(carry, action):
            state, value, discount = carry
            next_state = dynamics_model(state, action)
            reward = rm.get_reward(state, action)
            
            value += discount * reward
            discount *= self.discount
            return (next_state, value, discount), None

        # Initialize value estimation
        init_carry = (state, jnp.zeros(state.shape[0]), jnp.ones(state.shape[0]))
        (final_state, value, final_discount), _ = jax.lax.scan(
            step_fn, init_carry, actions.transpose(1, 0, 2)
        )
        
        # Get final action from actor for final state value estimation
        final_actions, _ = sac_agent.sample_actions_jit(final_state)
        q_values = sac_agent.critic.apply_fn(
            {"params": sac_agent.critic.params},
            final_state,
            final_actions
        )
        final_value = jnp.min(q_values, axis=0)  # Conservative estimate using min Q-value
        value += final_discount * final_value
        
        return value

    @partial(jax.jit, static_argnames=('self', 'dynamics_model'))
    def plan_step(self, key: jnp.ndarray, sac_agent, dynamics_model, rm, state: jnp.ndarray, 
                  prev_mean: jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Plan step without training noise."""
        mean = jnp.zeros((self.horizon, self.action_dim))
        std = 2.0 * jnp.ones((self.horizon, self.action_dim))
        
        if prev_mean is not None:
            mean = mean.at[:-1].set(prev_mean[1:])
        
        policy_state = jnp.repeat(state[None], self.num_policy_traj, axis=0)
        
        def policy_step(carry, _):
            curr_state, key = carry
            key, action_key = jax.random.split(key)
            action = sac_agent.eval_actions_jit(curr_state)  # Use jitted version
            next_state = dynamics_model(curr_state, action)
            return (next_state, key), action

        (_, _), policy_actions = jax.lax.scan(
            policy_step,
            (policy_state, key),
            None,
            length=self.horizon
        )
        
        def cem_iter(carry, _):
            mean, std, key = carry
            key, sample_key = jax.random.split(key)
            
            # Sample actions
            sample_actions = mean[:, None] + std[:, None] * jax.random.normal(
                sample_key,
                (self.horizon, self.num_sample_traj, self.action_dim)
            )
            sample_actions = jnp.clip(sample_actions, -0.999, 0.999)
            
            # Combine sampled and policy actions
            actions = jnp.concatenate([sample_actions, policy_actions], axis=1)
            
            # Evaluate actions
            imagine_returns = self.estimate_value(
                sac_agent, 
                dynamics_model, 
                rm,
                jnp.repeat(state[None], actions.shape[1], axis=0),
                actions.transpose(1, 0, 2)  # (batch, horizon, action_dim)
            )
            
            # Select elites
            elite_idx = jnp.argsort(imagine_returns)[-self.num_elites:]
            elite_values = imagine_returns[elite_idx]
            elite_actions = actions[:, elite_idx]  # (horizon, num_elites, action_dim)
            
            # Weighted aggregation of elite plans
            scores = jax.nn.softmax(self.cem_temperature * (elite_values - elite_values.max()))
            scores = scores.reshape(1, -1, 1)
            
            new_mean = (scores * elite_actions).sum(axis=1)
            new_std = jnp.sqrt(((elite_actions - new_mean[:, None]) ** 2 * scores).sum(axis=1))
            
            # Update distribution
            mean = self.cem_momentum * mean + (1 - self.cem_momentum) * new_mean
            std = jnp.clip(new_std, self.min_std, 2.0)
            
            return (mean, std, key), (elite_actions, scores)
        
        # Run CEM iterations
        (mean, std, key), (elite_actions, scores) = jax.lax.scan(
            cem_iter,
            (mean, std, key),
            None,
            length=self.cem_iter
        )
        
        # Sample action for MPC
        key, action_key = jax.random.split(key)
        
        # Get scores from last iteration and ensure proper shape
        scores = scores[-1]  # Get last iteration's scores
        scores = scores.reshape(-1)  # Flatten to 1D array
        
        # Sample index using the scores as probabilities
        action_idx = jax.random.choice(
            action_key,
            self.num_elites,  # number of elements to choose from
            shape=(1,),       # output shape
            p=scores         # probabilities (must sum to 1 and match length of array)
        )[0]
        
        # Get the selected action
        action = elite_actions[-1, 0, action_idx]  # Use last iteration's actions
        
        return action, mean, std[0]

    def plan(self, key: jnp.ndarray, sac_agent, dynamics_model, rm, state: jnp.ndarray, 
             prev_mean: jnp.ndarray = None, is_train: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Plan actions using CEM."""
        action, mean, std = self.plan_step(key, sac_agent, dynamics_model, rm, state, prev_mean)
        
        if is_train:
            key, noise_key = jax.random.split(key)
            action += std * jax.random.normal(noise_key, action.shape)
        
        return jnp.clip(action, -0.999, 0.999), mean
