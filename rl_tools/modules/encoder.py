import functools
from typing import Callable, Type

from einops import rearrange
import flax.linen as nn
import gymnasium.spaces as spaces
import jax
import jax.numpy as jnp

from rl_tools.modules.modules import conv_layer, MLP


class VisionEncoder(nn.Module):
    rearrange_pattern: str
    preprocess_fn: Callable = None

    @nn.compact
    def __call__(self, x: jax.Array):
        x = x.astype(jnp.float32)
        x = rearrange(x, self.rearrange_pattern)
        if self.preprocess_fn is not None:
            x = self.preprocess_fn(x)

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
    preprocess_fn: Callable = None

    @nn.compact
    def __call__(self, x: jax.Array):
        x = x.astype(jnp.float32)
        if self.preprocess_fn is not None:
            x = self.preprocess_fn(x)
        return MLP([256, 256], nn.relu, nn.relu)(x)


def encoder_factory(
    observation_space: spaces.Space,
    *,
    rearrange_pattern: str = "b h w c -> b h w c",
    preprocess_fn: Callable = None,
) -> Type[nn.Module]:
    if len(observation_space.shape) == 1:
        return functools.partial(VectorEncoder, preprocess_fn=preprocess_fn)
    elif len(observation_space.shape) == 3:
        return functools.partial(
            VisionEncoder,
            rearrange_pattern=rearrange_pattern,
            preprocess_fn=preprocess_fn,
        )
    else:
        raise NotImplementedError
