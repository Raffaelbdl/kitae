from typing import Callable, Iterable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from rl.types import Params


class PassThrough(nn.Module):
    @nn.compact
    def __call__(self, *x: Iterable[jax.Array]):
        return x


class MLP(nn.Module):
    layers: list[int]
    activation: Callable
    final_activation: Callable

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for l in self.layers[:-1]:
            x = self.activation(nn.Dense(l)(x))
        return self.final_activation(nn.Dense(self.layers[-1])(x))


def conv_layer(
    features: int,
    kernel_size: int,
    strides: int,
    kernel_init_std: float = np.sqrt(2.0),
    bias_init_cst: float = 0.0,
) -> nn.Conv:
    return nn.Conv(
        features,
        (kernel_size, kernel_size),
        strides,
        padding="VALID",
        kernel_init=nn.initializers.orthogonal(kernel_init_std),
        bias_init=nn.initializers.constant(bias_init_cst),
    )


def init_params(
    key: jax.Array,
    module: nn.Module,
    input_shapes: tuple[int] | list[tuple[int]],
    tabulate: bool,
) -> Params:
    if not isinstance(input_shapes, list):
        input_shapes = [input_shapes]

    dummy_inputs = [jnp.ones((1,) + shape) for shape in input_shapes]
    params = module.init(key, *dummy_inputs)

    if tabulate:
        tabulate_fn = nn.tabulate(
            module, key, compute_flops=True, compute_vjp_flops=True
        )
        print(tabulate_fn(*dummy_inputs))

    return params


class IndependentVariable(nn.Module):
    """Class for independent learnable variables.

    Used in SAC as the temperature coefficient.
    """

    name: str
    init_fn: Callable
    shape: tuple[int]

    @nn.compact
    def __call__(self) -> jax.Array:
        return self.param(self.name, self.init_fn, self.shape)
