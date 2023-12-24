import distrax as dx
from flax import linen as nn
import jax
from jax import numpy as jnp


class PolicyOutput(nn.Module):
    num_outputs: int

    @nn.compact
    def __call__(self, x: jax.Array) -> dx.Distribution:
        ...


class PolicyNormalOutput(PolicyOutput):
    num_outputs: int

    @nn.compact
    def __call__(self, x: jax.Array) -> dx.Normal:
        loc = nn.Dense(
            features=self.num_outputs,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        log_scale = jnp.broadcast_to(
            self.param("log_std", nn.initializers.zeros, (1, self.num_outputs)),
            loc.shape,
        )

        return dx.Normal(loc, jnp.exp(log_scale))


class PolicyStandardNormalOutput(PolicyOutput):
    num_outputs: int

    @nn.compact
    def __call__(self, x: jax.Array) -> dx.Normal:
        loc = nn.Dense(
            features=self.num_outputs,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0),
        )(x)

        return dx.Normal(loc, jnp.ones_like(loc))
