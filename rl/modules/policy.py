import distrax as dx
from flax import linen as nn
import jax
from jax import numpy as jnp

from dx_tabulate import add_representer


class Policy(nn.Module):
    num_outputs: int

    @nn.compact
    def __call__(self, x: jax.Array) -> dx.Distribution:
        ...


class PolicyCategorical(Policy):
    num_outputs: int

    @nn.compact
    def __call__(self, x: jax.Array) -> dx.Distribution:
        logits = nn.Dense(
            features=self.num_outputs,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0),
        )(x)

        return dx.Categorical(logits)


class PolicyNormal(Policy):
    num_outputs: int

    def setup(self) -> None:
        add_representer(dx.Normal)

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


class PolicyNormalExternalStd(Policy):
    num_outputs: int
    action_scale: jax.Array
    action_bias: jax.Array

    def setup(self) -> None:
        add_representer(dx.Normal)

    @nn.compact
    def __call__(self, x: jax.Array, policy_noise: float | jax.Array) -> dx.Normal:
        loc = nn.Dense(
            features=self.num_outputs,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0),
        )(x)

        loc = nn.tanh(loc) * self.action_scale + self.action_bias
        return dx.Normal(loc, policy_noise)
