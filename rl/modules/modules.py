import functools
from typing import Callable, Type

import chex
from einops import rearrange
import flax
from flax import struct
from flax import linen as nn
from flax.training.train_state import TrainState
from gymnasium import spaces
import jax
from jax import numpy as jnp
import numpy as np

from rl.base import Params


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


class VisionEncoder(nn.Module):
    rearrange_pattern: str

    @nn.compact
    def __call__(self, x: jax.Array):
        x = x.astype(jnp.float32)
        x = rearrange(x, self.rearrange_pattern)

        x = conv_layer(32, 8, 4)(x)
        x = nn.relu(x)
        x = conv_layer(64, 4, 2)(x)
        x = nn.relu(x)
        x = conv_layer(64, 3, 1)(x)
        x = nn.relu(x)

        x = jnp.reshape(x, (x.shape[0], -1))
        x = nn.Dense(
            features=512,
            kernel_init=nn.initializers.orthogonal(2.0),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        return nn.relu(x)


class VectorEncoder(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array):
        x = x.astype(jnp.float32)
        x = nn.Dense(
            features=64,
            kernel_init=nn.initializers.orthogonal(np.sqrt(2.0)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        x = nn.tanh(x)
        x = nn.Dense(
            features=64,
            kernel_init=nn.initializers.orthogonal(np.sqrt(2.0)),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        return nn.tanh(x)


class PolicyOutput(nn.Module):
    num_outputs: int

    @nn.compact
    def __call__(self, x: jax.Array):
        return nn.Dense(
            features=self.num_outputs,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0),
        )(x)


class ValueOutput(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array):
        return nn.Dense(
            features=1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.constant(0.0),
        )(x)


def encoder_factory(
    observation_space: spaces.Space,
    *,
    rearrange_pattern: str = "b h w c -> b h w c",
) -> Type[nn.Module]:
    if len(observation_space.shape) == 1:
        return VectorEncoder
    elif len(observation_space.shape) == 3:
        return functools.partial(VisionEncoder, rearrange_pattern=rearrange_pattern)
    else:
        raise NotImplementedError


def modules_factory(
    observation_space: spaces.Space,
    action_space: spaces.Space,
    shared_encoder: bool,
    *,
    rearrange_pattern: str = "b h w c -> b h w c",
) -> dict[str, nn.Module]:
    """Returns a dict of modules:
    policy / value / encoder"""
    encoder = encoder_factory(observation_space, rearrange_pattern=rearrange_pattern)

    num_actions = (
        action_space.n
        if isinstance(action_space, spaces.Discrete)
        else action_space.shape[-1]
    )

    if shared_encoder:
        return {
            "policy": PolicyOutput(num_actions),
            "value": ValueOutput(),
            "encoder": encoder(),
        }

    class Policy(nn.Module):
        @nn.compact
        def __call__(self, x: jax.Array):
            x = encoder()(x)
            return PolicyOutput(num_actions)(x)

    class Value(nn.Module):
        @nn.compact
        def __call__(self, x: jax.Array):
            x = encoder()(x)
            return ValueOutput()(x)

    class PassThrough(nn.Module):
        @nn.compact
        def __call__(self, x: jax.Array):
            return x

    return {"policy": Policy(), "value": Value(), "encoder": PassThrough()}


def create_params(
    key: jax.Array,
    module: nn.Module,
    input_shape: tuple[int],
    tabulate: bool,
) -> Params:
    dummy_inputs = jnp.ones((1,) + input_shape)
    variables = module.init(key, dummy_inputs)

    if tabulate:
        tabulate_fn = nn.tabulate(
            module, key, compute_flops=True, compute_vjp_flops=True
        )
        print(tabulate_fn(dummy_inputs))

    if "params" in variables.keys():
        return variables["params"]
    return {}
