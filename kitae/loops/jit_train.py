import logging
from typing import Callable, NamedTuple

from gymnasium.spaces import Space, Box, Discrete
import jax
import jax.numpy as jnp
import numpy as np

from jrd_extensions import PRNGSequence

from kitae.agent import BaseAgent
from kitae.modules.pytree import AgentPyTree

from kitae.buffers.buffer import Experience
from kitae.buffers.jit_buffer import jBuffer, BufferState, buffer_factory
from kitae.interface import AlgoType
from kitae.types import PRNGKeyArray

try:
    from gymnax.environments.environment import Environment, EnvParams, EnvState
    from gymnax.environments.spaces import (
        Discrete as jDiscrete,
        Box as jBox,
        Space as jSpace,
    )
except ImportError:
    logging.error("Module 'gymnax' is not installed. Run `pip install kitae[gymnax]`")
    raise ImportError


class jAgent(NamedTuple):
    state: AgentPyTree
    explore_fn: Callable
    preprocess_fn: Callable
    update_step_fn: Callable
    algo_type: AlgoType


class jEnv(NamedTuple):
    env: Environment
    env_params: EnvParams
    n_envs: int
    n_agents: int


def make_j_agent(agent: BaseAgent) -> jAgent:
    return jAgent(
        state=agent.state,
        explore_fn=agent.explore_fn,
        preprocess_fn=agent.experience_pipeline.run,
        update_step_fn=agent.update_step_fn,
        algo_type=agent.algo_type,
    )


class RunningState(NamedTuple):
    observation: jax.Array
    env_state: EnvState
    agent_state: AgentPyTree
    buffer_state: BufferState
    key: jax.Array


def make_env_step(
    j_env: jEnv, explore_fn: Callable, return_info: bool = True
) -> Callable:
    env, env_params, n_envs, _ = j_env
    env_vstep = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def env_step_fn(state: RunningState, prev_transition: Experience):
        rng = PRNGSequence(state.key)

        action, log_prob = explore_fn(state.agent_state, next(rng), state.observation)

        step_keys = jax.random.split(next(rng), n_envs)
        next_obs, env_state, reward, done, info = env_vstep(
            step_keys, state.env_state, action, env_params
        )

        next_state = RunningState(
            next_obs, env_state, state.agent_state, state.buffer_state, next(rng)
        )
        transition = Experience(
            state.observation, action, reward, done, next_obs, log_prob
        )

        if return_info:
            return next_state, transition, info
        return next_state, transition

    return env_step_fn


def make_update(j_agent: jAgent, j_buffer: jBuffer) -> Callable:
    def should_update(buffer_state: BufferState) -> bool:
        if j_agent.algo_type == AlgoType.ON_POLICY:
            return buffer_state.index >= j_buffer.max_buffer_size
        if j_agent.algo_type == AlgoType.OFF_POLICY:
            return buffer_state.length >= j_buffer.sample_size
        return False

    def update_fn(
        agent_state: AgentPyTree, key: PRNGKeyArray, buffer_state: BufferState
    ) -> tuple[AgentPyTree, BufferState]:
        def _update_fn(
            a_state: AgentPyTree, k: PRNGKeyArray, b_state: BufferState
        ) -> tuple[AgentPyTree, BufferState]:
            k1, k2, k3 = jax.random.split(k, 3)

            b_state, sample = j_buffer.sample(b_state, k1)
            experience = j_agent.preprocess_fn(a_state, k2, sample)
            a_state, loss_dict = j_agent.update_step_fn(a_state, k3, experience)

            return a_state, b_state

        agent_state, buffer_state = jax.lax.cond(
            should_update(buffer_state),
            lambda state, k: _update_fn(state, k, buffer_state),
            lambda state, k: (state, buffer_state),
            agent_state,
            key,
        )

        return agent_state, buffer_state

    return update_fn


def make_interact_and_update(
    j_agent: jAgent, j_env: jEnv, j_buffer: jBuffer, *, debug: bool = False
) -> Callable:
    env_step_fn = jax.jit(make_env_step(j_env, j_agent.explore_fn, True))
    update_fn = make_update(j_agent, j_buffer)

    def interact_and_update_fn(
        running_state: RunningState, _: None
    ) -> tuple[RunningState, None]:
        rng = PRNGSequence(running_state.key)

        running_state, transition, info = env_step_fn(running_state, None)
        buffer_state = j_buffer.add(running_state.buffer_state, transition)
        agent_state, buffer_state = update_fn(
            running_state.agent_state, next(rng), buffer_state
        )

        if debug:

            def display_metrics(info):
                episode_returns = info["returned_episode_returns"][
                    info["returned_episode"]
                ]
                for v in episode_returns:
                    print(v)
                    break

            jax.debug.callback(display_metrics, info)

        running_state = RunningState(
            running_state.observation,
            running_state.env_state,
            agent_state,
            buffer_state,
            next(rng),
        )
        return running_state, None

    return interact_and_update_fn


def make_train(
    j_env: jEnv,
    j_agent: jAgent,
    max_buffer_size: int,
    sample_size: int,
    n_env_steps: int,
    debug: bool = False,
) -> Callable:
    env, env_params, n_envs, _ = j_env
    env_vreset = jax.vmap(env.reset, in_axes=(0, None))

    buffer = buffer_factory(
        j_agent.algo_type, max_buffer_size, j_env.n_envs, sample_size
    )

    interact_and_update = make_interact_and_update(j_agent, j_env, buffer, debug=debug)

    def train(agent_state: AgentPyTree, key: jax.Array) -> AgentPyTree:
        rng = PRNGSequence(key)

        # initialize buffer
        obs, env_state = env.reset(next(rng), env_params)
        action = env.action_space(env_params).sample(next(rng))
        next_obs, _, reward, done, _ = env.step(
            next(rng), env_state, action, env_params
        )
        buffer_state = buffer.init(
            Experience(
                observation=obs,
                action=action,
                reward=reward,
                done=done,
                next_observation=next_obs,
                log_prob=jnp.zeros((1,), dtype=jnp.float32),
            ),
        )

        reset_keys = jax.random.split(next(rng), n_envs)
        observation, env_state = env_vreset(reset_keys, env_params)
        running_state = RunningState(
            observation, env_state, agent_state, buffer_state, next(rng)
        )

        # for t in range(n_env_steps):
        #     running_state, _ = interact_and_update(running_state, None)
        running_state, _ = jax.lax.scan(
            interact_and_update, running_state, None, n_env_steps
        )

        return running_state.agent_state

    return train


def gymnax_to_gym(space: jSpace) -> Space:
    if isinstance(space, jBox):
        return Box(np.array(space.low), np.array(space.high), space.shape, space.dtype)
    if isinstance(space, jDiscrete):
        return Discrete(space.n)

    raise TypeError
