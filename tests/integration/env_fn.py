from typing import Any, SupportsFloat, Tuple, Union

import chex
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from gymnax.environments.spaces import Discrete as jDiscrete, Box as jBox
from gymnax.environments.environment import (
    Environment as jEnv,
    TEnvState,
    TEnvParams,
    EnvState,
)
import jax.numpy as jnp

from kitae.envs.make import wrap_single_env


class RandomEnv(gym.Env):
    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs = self.observation_space.sample()
        return obs, 1.0, False, False, {}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        obs = self.observation_space.sample()
        return obs, {}


class DiscreteEnv(RandomEnv):
    action_space = Discrete(10)
    observation_space = Box(-1.0, 1.0, (20,))


def make_discrete_env():
    return wrap_single_env(DiscreteEnv())


class ContinuousEnv(RandomEnv):
    action_space = Box(-1.0, 1.0, (10,))
    observation_space = Box(-1.0, 1.0, (20,))


def make_continuous_env():
    return wrap_single_env(ContinuousEnv())


class jRandomEnv(jEnv):
    def step_env(
        self,
        key: chex.PRNGKey,
        state: TEnvState,
        action: Union[int, float, chex.Array],
        params: TEnvParams,
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs = self.observation_space(params).sample(key)
        return obs, EnvState(state.time + 1), jnp.array(1.0), False, {}

    def reset_env(
        self, key: chex.PRNGKey, params: TEnvParams
    ) -> Tuple[chex.Array, TEnvState]:
        obs = self.observation_space(params).sample(key)
        return obs, EnvState(0)


class jDiscreteEnv(jRandomEnv):

    def action_space(self, params: Any):
        return jDiscrete(10)

    def observation_space(self, params: Any):
        high = jnp.ones((20,))
        low = -jnp.ones((20,))
        return jBox(low, high, (20,))


def make_j_discrete_env():
    return jDiscreteEnv()


class jContinuousEnv(jRandomEnv):

    def action_space(self, params: Any):
        high = jnp.ones((10,))
        low = -jnp.ones((10,))
        return jBox(low, high, (10,))

    def observation_space(self, params: Any):
        high = jnp.ones((20,))
        low = -jnp.ones((20,))
        return jBox(low, high, (20,))


def make_j_continuous_env():
    return jContinuousEnv()
