from typing import Any, Callable

from gymnasium import spaces
import flax.linen as nn
import jax
import jax.numpy as jnp

from rl import Params


def ensure_int(action: jax.Array | int) -> int:
    return int(action)
