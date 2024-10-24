from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames="padding")
def random_crop(key, img, padding):
    crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
    crop_from = jnp.concatenate([crop_from, jnp.zeros((2,), dtype=jnp.int32)])
    padded_img = jnp.pad(
        img, ((padding, padding), (padding, padding), (0, 0), (0, 0)), mode="edge"
    )
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=["pixel_key", "frozen"])
def batched_random_crop(key, obs, pixel_key, frozen=True, padding=4):
    imgs = obs[pixel_key]
    num_dims = len(imgs.shape)
    if num_dims == 6:
        A, B, C, D, E, F = imgs.shape
        imgs = imgs.reshape(A * B, C, D, E, F)
    keys = jax.random.split(key, imgs.shape[0])
    imgs = jax.vmap(random_crop, (0, 0, None))(keys, imgs, padding)
    if num_dims == 6:
        imgs = imgs.reshape(A, B, C, D, E, F)
    if frozen:
        return obs.copy(add_or_replace={pixel_key: imgs})
    else:
        obs[pixel_key] = imgs
        return obs
