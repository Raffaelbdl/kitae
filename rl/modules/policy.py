import distrax as dx
from flax import linen as nn
from gymnasium.spaces import Box, Discrete
import jax
from jax import numpy as jnp

from dx_tabulate import add_representer


class PolicyOutput(nn.Module):
    num_outputs: int

    @nn.compact
    def __call__(self, x: jax.Array) -> dx.Distribution: ...


class PolicyCategorical(PolicyOutput):
    num_outputs: int

    @nn.compact
    def __call__(self, x: jax.Array) -> dx.Distribution:
        logits = nn.Dense(
            features=self.num_outputs,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(0.0),
        )(x)

        return dx.Categorical(logits)


class PolicyNormal(PolicyOutput):
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


class PolicyNormalExternalStd(PolicyOutput):
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


class PolicyTanhNormal(PolicyOutput):
    num_outputs: int
    log_std_min: float
    log_std_max: float

    def setup(self) -> None:
        add_representer(dx.Normal)
        add_representer(dx.Transformed)

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
        log_scale = jnp.clip(log_scale, self.log_std_min, self.log_std_max)

        # base_dist = dx.MultivariateNormalDiag(loc, jnp.exp(log_scale))
        # return dx.Transformed(base_dist, dx.Tanh())
        base_dist = dx.Normal(loc, jnp.exp(log_scale))
        return dx.Transformed(base_dist, dx.Tanh())


def make_policy(encoder: nn.Module, policy_output: PolicyOutput) -> nn.Module:
    class Policy(nn.Module):
        def setup(self) -> None:
            self.encoder = encoder
            self.output = policy_output

        def __call__(self, x: jax.Array) -> dx.Distribution:
            return self.output(self.encoder(x))

    return Policy()


def policy_output_factory(action_space: Discrete) -> type[PolicyOutput]:
    if isinstance(action_space, Discrete):
        add_representer(dx.Categorical)
        return PolicyCategorical
    elif isinstance(action_space, Box):
        add_representer(dx.Normal)
        return PolicyNormal
    else:
        raise NotImplementedError
