from typing import Callable, Type, Iterable

from flax import linen as nn
from gymnasium import spaces
import jax
import jax.numpy as jnp

from rl_tools.modules.encoder import encoder_factory


class QValueDiscreteOutput(nn.Module):
    num_outputs: int

    @nn.compact
    def __call__(self, x: jax.Array):
        return nn.Dense(
            features=self.num_outputs,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.constant(0.0),
        )(x)


class QValueContinousOutput(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array, a: jax.Array):
        x = jnp.concatenate([x, a], axis=-1)
        x = nn.relu(nn.Dense(64)(x))
        return nn.Dense(
            features=1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.constant(0.0),
        )(x)


class DoubleQValueContinuousOutput(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array, a: jax.Array):
        return QValueContinousOutput()(x, a), QValueContinousOutput()(x, a)


def qvalue_factory(
    observation_space: spaces.Space,
    action_space: spaces.Space,
    *,
    rearrange_pattern: str = "b h w c -> b h w c",
    preprocess_fn: Callable = None,
) -> Type[nn.Module]:
    encoder = encoder_factory(
        observation_space,
        rearrange_pattern=rearrange_pattern,
        preprocess_fn=preprocess_fn,
    )

    if isinstance(action_space, spaces.Discrete):

        class QValue(nn.Module):
            def setup(self) -> None:
                self.encoder = encoder()

            @nn.compact
            def __call__(self, x: jax.Array) -> jax.Array:
                x = self.encoder(x)
                return nn.Dense(action_space.n)(x)

    elif isinstance(action_space, spaces.Box):

        class QValue(nn.Module):
            def setup(self) -> None:
                self.encoder = encoder()

            @nn.compact
            def __call__(self, x: jax.Array, a: jax.Array) -> jax.Array:
                if len(observation_space.shape) == 1:
                    x = self.encoder(jnp.concatenate([x, a], axis=-1))

                elif len(observation_space.shape) == 3:
                    x = self.encoder(x)
                    x = jnp.concatenate([x, a], axis=-1)

                else:
                    raise NotImplementedError

                return nn.Dense(1)(x)

    else:
        raise NotImplementedError

    return QValue


def make_double_q_value(q1: nn.Module, q2: nn.Module) -> nn.Module:
    class DoubleQValue(nn.Module):
        def setup(self) -> None:
            self.q1 = q1
            self.q2 = q2

        def __call__(self, *arrays: Iterable[jax.Array]) -> tuple[jax.Array, jax.Array]:
            return self.q1(*arrays), self.q2(*arrays)

    return DoubleQValue()
