"""Deep Q-Network (DQN)"""

from collections import namedtuple
from dataclasses import dataclass
from typing import Callable

import distrax as dx
import jax
import jax.numpy as jnp
import optax

from kitae.agent import OffPolicyAgent
from kitae.buffers.buffer import Experience
from kitae.config import AlgoConfig, AlgoParams

from kitae.operations.timesteps import compute_td_targets

from kitae.modules.modules import init_params
from kitae.modules.pytree import AgentPyTree, TrainState
from kitae.modules.qvalue import qvalue_factory

from kitae.types import Params, PRNGKeyArray, LossDict
from kitae.types import ExploreFn, ProcessExperienceFn, UpdateFn

DQN_tuple = namedtuple("DQN_tuple", ["observation", "action", "return_"])


class DQNState(AgentPyTree):
    qvalue_state: TrainState


@dataclass
class DQNParams(AlgoParams):
    """Deep Q-Network parameters."""

    exploration: float = 0.1
    gamma: float = 0.99
    skip_steps: int = 1  # > 0
    start_step: int = -1


def train_state_dqn_factory(
    key: PRNGKeyArray,
    config: AlgoConfig,
    *,
    preprocess_fn: Callable,
    tabulate: bool = False,
) -> DQNState:
    observation_shape = config.env_cfg.observation_space.shape

    qvalue = qvalue_factory(
        config.env_cfg.observation_space, config.env_cfg.action_space
    )()
    qvalue_state = TrainState.create(
        apply_fn=jax.jit(qvalue.apply),
        params=init_params(key, qvalue, [observation_shape], tabulate),
        target_params=init_params(key, qvalue, [observation_shape], False),
        tx=optax.adam(config.update_cfg.learning_rate),
    )
    return DQNState(qvalue_state=qvalue_state)


def explore_factory(config: AlgoConfig) -> ExploreFn:

    def explore_fn(
        state: DQNState,
        key: PRNGKeyArray,
        observations: jax.Array,
        *,
        exploration: float,
    ) -> tuple[jax.Array, jax.Array]:

        all_qvalues = state.qvalue_state.apply_fn(
            state.qvalue_state.params, observations
        )
        dists = dx.EpsilonGreedy(all_qvalues, exploration)
        actions, log_probs = dists.sample_and_log_prob(seed=key)

        return actions, log_probs

    return jax.jit(explore_fn)


def process_experience_factory(config: AlgoConfig) -> ProcessExperienceFn:
    algo_params: DQNParams = config.algo_params

    def process_experience_fn(
        state: DQNState, key: PRNGKeyArray, experience: Experience
    ) -> DQN_tuple:

        all_next_qvalues = state.qvalue_state.apply_fn(
            state.qvalue_state.params, experience.next_observation
        )
        next_qvalues = jnp.max(all_next_qvalues, axis=-1, keepdims=True)

        discounts = algo_params.gamma * (1.0 - experience.done[..., None])
        returns = compute_td_targets(
            experience.reward[..., None], discounts, next_qvalues
        )
        actions = experience.action[..., None]

        return DQN_tuple(experience.observation, actions, returns)

    return jax.jit(process_experience_fn)


def update_qvalue_factory(config: AlgoConfig) -> UpdateFn:

    def update_qvalue_fn(
        state: DQNState, key: PRNGKeyArray, batch: DQN_tuple
    ) -> tuple[DQNState, LossDict]:

        def loss_fn(params: Params):
            all_qvalues = state.qvalue_state.apply_fn(params, batch.observation)
            qvalues = jnp.take_along_axis(all_qvalues, batch.action, axis=-1)
            loss = jnp.mean(optax.l2_loss(qvalues, batch.return_))
            return loss, {"loss_qvalue": loss}

        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.qvalue_state.params
        )
        state.qvalue_state = state.qvalue_state.apply_gradients(grads=grads)

        return state, info

    return jax.jit(update_qvalue_fn)


def update_step_factory(config: AlgoConfig) -> Callable:

    update_qvalue_fn = update_qvalue_factory(config)

    def update_step_fn(
        state: DQNState, key: PRNGKeyArray, batch: DQN_tuple
    ) -> tuple[DQNState, LossDict]:

        return update_qvalue_fn(state, key, batch)

    return jax.jit(update_step_fn)


class DQN(OffPolicyAgent):
    """
    Deep Q-Network (DQN)
    Paper : https://arxiv.org/abs/1312.5602
    """

    def __init__(
        self,
        run_name: str,
        config: AlgoConfig,
        *,
        preprocess_fn: Callable = None,
        tabulate: bool = False,
    ):
        self.algo_params: DQNParams = config.algo_params

        super().__init__(
            run_name,
            config,
            train_state_dqn_factory,
            explore_factory,
            process_experience_factory,
            update_step_factory,
            preprocess_fn=preprocess_fn,
            tabulate=tabulate,
            experience_type=Experience,
        )

    def select_action(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        keys = self.interact_keys(observation)
        action, zeros = self.explore_fn(self.state, keys, observation, exploration=0.0)

        return action, zeros

    def explore(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        keys = self.interact_keys(observation)

        action, zeros = self.explore_fn(
            self.state, keys, observation, exploration=self.algo_params.exploration
        )

        return action, zeros
