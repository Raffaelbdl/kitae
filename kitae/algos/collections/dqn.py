"""Deep Q-Network (DQN)"""

from collections import namedtuple
from dataclasses import dataclass
from typing import Callable

import distrax as dx
import jax
import jax.numpy as jnp
import optax


from kitae.base import OffPolicyAgent
from kitae.config import AlgoConfig, AlgoParams
from kitae.types import Params

from kitae.buffer import Experience
from kitae.timesteps import compute_td_targets

from kitae.modules.modules import init_params
from kitae.modules.train_state import TrainState
from kitae.modules.qvalue import qvalue_factory

from kitae.pytree import AgentPyTree

DQN_tuple = namedtuple("DQN_tuple", ["observation", "action", "return_"])
NO_EXPLORATION = 0.0


class DQNState(AgentPyTree):
    qvalue: TrainState


@dataclass
class DQNParams(AlgoParams):
    """
    Deep Q-Network parameters

    Parameters:
        exploration: The exploration coefficient of the epsilon-greedy policy.
        gamma: The discount factor.
        skip_step: The numbers of steps skipped when training.
    """

    exploration: float
    gamma: float
    skip_steps: int
    start_step: int = -1


def train_state_dqn_factory(
    key: jax.Array,
    config: AlgoConfig,
    *,
    preprocess_fn: Callable,
    tabulate: bool = False,
) -> DQNState:
    observation_shape = config.env_cfg.observation_space.shape

    qvalue = qvalue_factory(
        config.env_cfg.observation_space, config.env_cfg.action_space
    )()
    return DQNState(
        TrainState.create(
            apply_fn=jax.jit(qvalue.apply),
            params=init_params(key, qvalue, [observation_shape], tabulate),
            target_params=init_params(key, qvalue, [observation_shape], False),
            tx=optax.adam(config.update_cfg.learning_rate),
        )
    )


def explore_factory(config: AlgoConfig) -> Callable:
    @jax.jit
    def explore_fn(
        qvalue_state: TrainState,
        key: jax.Array,
        observations: jax.Array,
        exploration: float,
    ) -> tuple[jax.Array, jax.Array]:
        all_qvalues = qvalue_state.apply_fn(qvalue_state.params, observations)
        actions, log_probs = dx.EpsilonGreedy(
            all_qvalues, exploration
        ).sample_and_log_prob(seed=key)

        return actions, log_probs

    return explore_fn


def process_experience_factory(config: AlgoConfig) -> Callable:
    algo_params = config.algo_params

    @jax.jit
    def process_experience_fn(
        dqn_state: DQNState,
        key: jax.Array,
        experience: Experience,
    ) -> tuple[jax.Array, ...]:

        all_next_qvalues = dqn_state.qvalue.apply_fn(
            dqn_state.qvalue.params, experience.next_observation
        )
        next_qvalues = jnp.max(all_next_qvalues, axis=-1, keepdims=True)

        discounts = algo_params.gamma * (1.0 - experience.done[..., None])
        returns = compute_td_targets(
            experience.reward[..., None], discounts, next_qvalues
        )
        actions = experience.action[..., None]

        return (experience.observation, actions, returns)

    return process_experience_fn


def update_step_factory(config: AlgoConfig) -> Callable:

    @jax.jit
    def update_qvalue_fn(qvalue_state: TrainState, batch: DQN_tuple):

        def loss_fn(params: Params):
            all_qvalues = qvalue_state.apply_fn(params, batch.observation)
            qvalues = jnp.take_along_axis(all_qvalues, batch.action, axis=-1)
            loss = jnp.mean(optax.l2_loss(qvalues, batch.return_))
            return loss, {"loss_qvalue": loss}

        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            qvalue_state.params
        )
        qvalue_state = qvalue_state.apply_gradients(grads=grads)

        return qvalue_state, info

    @jax.jit
    def update_step_fn(
        state: DQNState,
        key: jax.Array,
        experiences: tuple[jax.Array, ...],
    ) -> tuple[TrainState, dict]:
        batch = DQN_tuple(*experiences)
        state.qvalue, info = update_qvalue_fn(state.qvalue, batch)
        return state, info

    return update_step_fn


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

        self.algo_params = self.config.algo_params

    def select_action(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        keys = self.interact_keys(observation)

        action, zeros = self.explore_fn(
            self.state.qvalue, keys, observation, exploration=NO_EXPLORATION
        )
        return action, zeros

    def explore(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        keys = self.interact_keys(observation)

        action, zeros = self.explore_fn(
            self.state.qvalue,
            keys,
            observation,
            exploration=self.algo_params.exploration,
        )
        return action, zeros
