from functools import partial
from typing import Callable, Self

import jax
from jax import Array, numpy as jnp
from joblib import Memory

from src.data import DIR_ROOT

type KernelFn = Callable[[Array, Array], Array]

DIR_CACHE = DIR_ROOT / "cache"
DIR_CACHE.mkdir(exist_ok=True, parents=True)


def gaussian_transformation(squared_distance: Array, sigma: float) -> Array:
    return jnp.exp(-squared_distance / (2 * sigma ** 2))


def gaussian_kernel_fn(x_1: Array, x_2: Array, sigma: float) -> Array:
    difference = x_1 - x_2
    return gaussian_transformation(jnp.dot(difference, difference), sigma)


class Kernel:
    __fn: KernelFn

    @classmethod
    def gaussian(cls, sigma: float) -> Self:
        return cls(fn=partial(gaussian_kernel_fn, sigma=sigma))

    def __init__(self, fn: KernelFn):
        self.__fn = fn

    @partial(jax.jit, static_argnums={0})
    def __call__(self, x_1: Array, x_2: Array) -> Array:
        @partial(jnp.vectorize, signature="(n),(n)->()")
        def vectorized(x_1_: Array, x_2_: Array) -> Array:
            return self.__fn(x_1_, x_2_)

        return vectorized(x_1, x_2)

    @partial(jax.jit, static_argnums={0})
    def kernel_matrix(self, xs_1: Array, xs_2: Array) -> Array:
        if xs_1.ndim < 2:
            raise ValueError("Input arrays must have at least 2 dimensions.")
        if xs_2.ndim < 2:
            raise ValueError("Input arrays must have at least 2 dimensions.")

        return self(xs_1[..., None, :], xs_2[..., None, :, :])


def masked_kernel_matrix(kernel: Kernel, xs_1: Array, xs_2: Array, mask_1: Array, mask_2: Array) -> Array:
    kernel_matrix = kernel.kernel_matrix(xs_1, xs_2)
    kernel_matrix = kernel_matrix * ~mask_1[..., None] * ~mask_2[..., None, :]
    return kernel_matrix


def kme_dot(kernel: Kernel, xs_1: Array, xs_2: Array, mask_1: Array, mask_2: Array) -> Array:
    kernel_matrix = masked_kernel_matrix(kernel, xs_1, xs_2, mask_1, mask_2)
    normalizing_constant = (~mask_1).sum() * (~mask_2).sum()
    return kernel_matrix.sum() / normalizing_constant


memory_pairwise_kme_dot = Memory(location=DIR_CACHE / "pairwise_kme_dot")


@memory_pairwise_kme_dot.cache
def pairwise_kme_dot(
        kernel: Kernel, xs_batch_1: Array, xs_batch_2: Array, mask_batch_1: Array, mask_batch_2: Array
) -> Array:
    if xs_batch_1.ndim != 3:
        raise ValueError("Input array must have 3 dimensions.")
    if xs_batch_2.ndim != 3:
        raise ValueError("Input array must have 3 dimensions.")
    if mask_batch_1.shape != xs_batch_1.shape[:-1]:
        raise ValueError("Mask shape must match the first two dimensions of xs_batch.")
    if mask_batch_2.shape != xs_batch_2.shape[:-1]:
        raise ValueError("Mask shape must match the first two dimensions of xs_batch.")

    @partial(jax.vmap, in_axes=(None, None, None, 0, 0))
    def batch_kme_dp(kernel_: Kernel, xs_1: Array, mask_1: Array, xs_2: Array, mask_2: Array) -> Array:
        return kme_dot(kernel_, xs_1, xs_2, mask_1, mask_2)

    return jax.lax.map(
        f=lambda inp_1: batch_kme_dp(kernel, *inp_1, xs_batch_2, mask_batch_2),
        xs=(xs_batch_1, mask_batch_1)
    )


def pairwise_squared_mmd(
        kernel: Kernel, xs_batch_1: Array, xs_batch_2: Array, mask_batch_1: Array, mask_batch_2: Array,
        cached: bool = True
) -> Array:
    if cached:
        kme_dps = pairwise_kme_dot(kernel, xs_batch_1, xs_batch_2, mask_batch_1, mask_batch_2)
    else:
        kme_dps = pairwise_kme_dot.func(kernel, xs_batch_1, xs_batch_2, mask_batch_1, mask_batch_2)

    @partial(jax.vmap, in_axes=(None, 0, 0))
    def batch_kme_norm(kernel_: Kernel, xs: Array, mask: Array) -> Array:
        return kme_dot(kernel_, xs, xs, mask, mask)

    norms_1 = batch_kme_norm(kernel, xs_batch_1, mask_batch_1)
    norms_2 = batch_kme_norm(kernel, xs_batch_2, mask_batch_2)

    squared_mmds = norms_1[..., None] + norms_2[..., None, :] - 2 * kme_dps

    return jnp.clip(squared_mmds, min=0)  # clip to prevent numerical issues
