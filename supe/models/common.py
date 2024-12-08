from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct


class TrainState(struct.PyTreeNode):
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: Dict
    tx: Optional[optax.GradientTransformation] = struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, apply_fn, params, tx):
        opt_state = tx.init(params) if tx is not None else None
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
        )


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activate_final: bool = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = nn.relu(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        return x


class Model:
    state: TrainState

    def _update_step(self, loss_fn: Callable, state: TrainState, *args, **kwargs) -> Tuple[TrainState, Dict[str, float]]:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, info), grads = grad_fn(state.params, *args, **kwargs)

        # Update parameters
        new_state = state.apply_gradients(grads=grads) if state.tx is not None else state

        info.update({"grad_norm": optax.global_norm(grads)})
        return new_state, info 