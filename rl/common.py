from typing import Any, Callable

from gymnasium import spaces
import flax.linen as nn
import jax
import jax.numpy as jnp

from rl import Params


def ensure_int(action: jax.Array | int) -> int:
    return int(action)


def create_params(
    key: jax.Array,
    module: nn.Module,
    observation_shape: tuple[int],
) -> Params:
    dummy_inputs = jnp.ones((1,) + observation_shape)
    return module.init(key, dummy_inputs)["params"]
