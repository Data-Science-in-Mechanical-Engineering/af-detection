import jax.numpy as jnp
from jax import Array


def rri_length(peak_indices: Array, mask: Array) -> tuple[Array, Array]:
    assert peak_indices.ndim >= 1
    assert peak_indices.shape == mask.shape

    mask = mask[..., 1:]
    rris = (peak_indices[..., 1:] - peak_indices[..., :-1]) * ~mask

    return rris, mask


def normalized_rri_length(peak_indices: Array, mask: Array) -> tuple[Array, Array]:
    rris, mask = rri_length(peak_indices, mask)
    means = jnp.sum(rris, axis=-1, keepdims=True) / jnp.sum(~mask, axis=-1, keepdims=True)
    return rris / means, mask


def features(peak_indices: Array, mask: Array) -> tuple[Array, Array]:
    rris, mask = normalized_rri_length(peak_indices, mask)
    rris = rris[..., None]
    return rris, mask
