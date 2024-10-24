from typing import Dict, Optional, Type, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from supe.networks.encoders import D4PGEncoder

default_init = nn.initializers.xavier_uniform


class PixelMultiplexer(nn.Module):
    pixel_encoder: D4PGEncoder
    network_cls: Type[nn.Module]
    stop_gradient: bool = False
    latent_dim: int = 50

    @nn.compact
    def __call__(
        self,
        observations: Union[FrozenDict, Dict],
        actions: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        obs_pixels = observations["pixels"]

        obs_pixels = obs_pixels.astype(jnp.float32) / 255.0

        obs_pixels = jnp.reshape(
            obs_pixels, (*obs_pixels.shape[:-2], -1)
        )  # remove stacking dimension

        x = self.pixel_encoder(obs_pixels)

        if self.stop_gradient:
            x = jax.lax.stop_gradient(x)

        x = nn.Dense(self.latent_dim, kernel_init=default_init())(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        if "state" in observations:
            y = nn.Dense(self.latent_dim, kernel_init=default_init())(
                observations["state"]
            )
            y = nn.LayerNorm()(y)
            y = nn.tanh(y)

            observations = jnp.concatenate([x, y], axis=-1)
        else:
            observations = x

        if actions is None:
            return self.network_cls()(observations)
        else:
            return self.network_cls()(observations, actions)


def share_encoder(source, target):
    replacers = {}
    for k, v in source.params.items():
        if "encoder" in k:
            replacers[k] = v
    if "seq_encoder" in replacers:
        del replacers["seq_encoder"]

    assert "pixel_encoder" in replacers
    assert len(replacers) == 1, f"replacers: {replacers}"
    assert "pixel_encoder" in target.params, f"target.params: {target.params}"

    new_params = FrozenDict(target.params).copy(add_or_replace=replacers)
    target = target.replace(params=new_params)
    return target
