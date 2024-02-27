"""Proximal Policy Optimization (PPO)"""

from dataclasses import dataclass
from typing import Callable

import distrax as dx
import flax.linen as nn
from gymnasium import spaces
import jax
import jax.numpy as jnp
import numpy as np
import optax

from rl_tools.base import OnPolicyAgent, EnvType, EnvProcs, AlgoType
from rl_tools.callbacks.callback import Callback
from rl_tools.config import AlgoConfig, AlgoParams
from rl_tools.types import Params, GymEnv, EnvPoolEnv

from rl_tools.buffer import Experience
from rl_tools.distribution import get_log_probs
from rl_tools.loss import loss_policy_ppo, loss_value_clip
from rl_tools.timesteps import calculate_gaes_targets

from rl_tools.train import train

from rl_tools.algos.factory import AlgoFactory
from rl_tools.modules.encoder import encoder_factory
from rl_tools.modules.modules import PassThrough, init_params
from rl_tools.modules.optimizer import linear_learning_rate_schedule
from rl_tools.modules.policy import policy_output_factory
from rl_tools.modules.train_state import PolicyValueTrainState, TrainState
from rl_tools.modules.value import ValueOutput


@dataclass
class PPOParams(AlgoParams):
    """
    Proximal Policy Optimization parameters.

    Parameters:
        gamma: The discount factor.
        _lambda: The factor for Generalized Advantage Estimator.
        clip_eps: The clipping range for update.
        entropy_coef: The loss coefficient of the entropy loss.
        value_coef: The loss coefficient of the value loss.
        normalize:  If true, advantages are normalized.
    """

    gamma: float
    _lambda: float
    clip_eps: float
    entropy_coef: float
    value_coef: float
    normalize: bool


def train_state_ppo_factory(
    key: jax.Array,
    config: AlgoConfig,
    *,
    rearrange_pattern: str,
    preprocess_fn: Callable,
    tabulate: bool = False,
) -> PolicyValueTrainState:

    key1, key2, key3 = jax.random.split(key, 3)
    observation_shape = config.env_cfg.observation_space.shape
    hidden_shape = (512,) if len(observation_shape) == 3 else (256,)
    action_shape = config.env_cfg.action_space.shape

    encoder = encoder_factory(
        config.env_cfg.observation_space,
        rearrange_pattern=rearrange_pattern,
        preprocess_fn=preprocess_fn,
    )
    policy_output = policy_output_factory(config.env_cfg.action_space)
    n_actions = (
        config.env_cfg.action_space.n
        if isinstance(config.env_cfg.action_space, spaces.Discrete)
        else action_shape[-1]
    )

    if config.update_cfg.shared_encoder:
        policy = policy_output(n_actions)
        value = ValueOutput()
        encoder = encoder()
        input_shape = hidden_shape
    else:
        policy = nn.Sequential([encoder(), policy_output(n_actions)])
        value = nn.Sequential([encoder(), ValueOutput()])
        encoder = PassThrough()
        input_shape = observation_shape

    learning_rate = config.update_cfg.learning_rate
    if config.update_cfg.learning_rate_annealing:
        learning_rate = linear_learning_rate_schedule(
            learning_rate,
            0.0,
            n_envs=config.env_cfg.n_envs,
            n_env_steps=config.train_cfg.n_env_steps,
            max_buffer_size=config.update_cfg.max_buffer_size,
            batch_size=config.update_cfg.batch_size,
            num_epochs=config.update_cfg.n_epochs,
        )
    tx = optax.chain(
        optax.clip_by_global_norm(config.update_cfg.max_grad_norm),
        optax.adam(learning_rate, eps=1e-5),
    )

    policy_state = TrainState.create(
        apply_fn=jax.jit(policy.apply),
        params=init_params(key1, policy, [input_shape], tabulate),
        tx=tx,
    )
    value_state = TrainState.create(
        apply_fn=jax.jit(value.apply),
        params=init_params(key2, value, [input_shape], tabulate),
        tx=tx,
    )
    encoder_state = TrainState.create(
        apply_fn=jax.jit(encoder.apply),
        params=init_params(key3, encoder, [observation_shape], tabulate),
        tx=tx,
    )

    return PolicyValueTrainState(
        policy_state=policy_state, value_state=value_state, encoder_state=encoder_state
    )


def explore_factory(config: AlgoConfig) -> Callable:
    @jax.jit
    def fn(
        ppo_state: PolicyValueTrainState, key: jax.Array, observations: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        hiddens = ppo_state.encoder_state.apply_fn(
            ppo_state.encoder_state.params, observations
        )
        if not isinstance(hiddens, tuple):
            hiddens = (hiddens,)
        dists: dx.Distribution = ppo_state.policy_state.apply_fn(
            ppo_state.policy_state.params, *hiddens
        )
        return dists.sample_and_log_prob(seed=key)

    return fn


def process_experience_factory(config: AlgoConfig) -> Callable:
    algo_params = config.algo_params

    @jax.jit
    def fn(ppo_state: PolicyValueTrainState, key: jax.Array, experience: Experience):
        all_obs = jnp.concatenate(
            [experience.observation, experience.next_observation[-1:]], axis=0
        )
        all_hiddens = ppo_state.encoder_state.apply_fn(
            ppo_state.encoder_state.params, all_obs
        )
        if not isinstance(all_hiddens, tuple):
            all_hiddens = (all_hiddens,)
        all_values = ppo_state.value_state.apply_fn(
            ppo_state.value_state.params, *all_hiddens
        )

        values = all_values[:-1]
        next_values = all_values[1:]

        not_dones = 1.0 - experience.done[..., None]
        discounts = algo_params.gamma * not_dones

        rewards = experience.reward[..., None]
        gaes, targets = calculate_gaes_targets(
            values,
            next_values,
            discounts,
            rewards,
            algo_params._lambda,
            algo_params.normalize,
        )

        return (
            experience.observation,
            experience.action,
            experience.log_prob,
            gaes,
            targets,
            values,
        )

    return fn


def update_step_factory(config: AlgoConfig) -> Callable:
    algo_params = config.algo_params

    def update_policy_value_fn(
        ppo_state: PolicyValueTrainState, batch: tuple[jax.Array, ...]
    ):

        observations, actions, log_probs_old, gaes, targets, values_old = batch

        def loss_fn(params):
            hiddens = ppo_state.encoder_state.apply_fn(params["encoder"], observations)
            if not isinstance(hiddens, tuple):
                hiddens = (hiddens,)
            dists = ppo_state.policy_state.apply_fn(params["policy"], *hiddens)
            log_probs, _log_probs_old = get_log_probs(dists, actions, log_probs_old)
            loss_policy, info_policy = loss_policy_ppo(
                dists,
                log_probs,
                _log_probs_old,
                gaes,
                algo_params.clip_eps,
                algo_params.entropy_coef,
            )

            values = ppo_state.value_state.apply_fn(params["value"], *hiddens)
            loss_value, info_value = loss_value_clip(
                values, targets, values_old, algo_params.clip_eps
            )

            loss = loss_policy + algo_params.value_coef * loss_value
            info = info_policy | info_value
            info["total_loss"] = loss

            return loss, info

        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            {
                "policy": ppo_state.policy_state.params,
                "value": ppo_state.value_state.params,
                "encoder": ppo_state.encoder_state.params,
            },
        )

        ppo_state.policy_state = ppo_state.policy_state.apply_gradients(
            grads=grads["policy"]
        )
        ppo_state.value_state = ppo_state.value_state.apply_gradients(
            grads=grads["value"]
        )
        ppo_state.encoder_state = ppo_state.encoder_state.apply_gradients(
            grads=grads["encoder"]
        )
        return ppo_state, loss, info

    @jax.jit
    def update_step_fn(
        ppo_state: PolicyValueTrainState,
        key: jax.Array,
        experiences: tuple[jax.Array, ...],
    ):
        num_elems = experiences[0].shape[0]
        iterations = num_elems // config.update_cfg.batch_size
        inds = jax.random.permutation(key, num_elems)[
            : iterations * config.update_cfg.batch_size
        ]

        batches = jax.tree_util.tree_map(
            lambda x: x[inds].reshape(
                (iterations, config.update_cfg.batch_size) + x.shape[1:]
            ),
            experiences,
        )

        loss = 0.0
        for batch in zip(*batches):
            ppo_state, l, info = update_policy_value_fn(ppo_state, batch)
            loss += l
        loss /= iterations

        return ppo_state, info

    return update_step_fn


class PPO(OnPolicyAgent):
    """
    Proximal Policy Optimization (PPO)
    Paper : https://arxiv.org/abs/1707.06347
    Implementation details : https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    """

    def __init__(
        self,
        config: AlgoConfig,
        *,
        rearrange_pattern: str = "b h w c -> b h w c",
        preprocess_fn: Callable = None,
        run_name: str = None,
        tabulate: bool = False,
    ):
        AlgoFactory.intialize(
            self,
            config,
            train_state_ppo_factory,
            explore_factory,
            process_experience_factory,
            update_step_factory,
            rearrange_pattern=rearrange_pattern,
            preprocess_fn=preprocess_fn,
            run_name=run_name,
            tabulate=tabulate,
            experience_type=Experience,
        )

    def select_action(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        return self.explore(observation)

    def explore(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        # TODO remove this and put it in the factory !
        keys = (
            {a: self.nextkey() for a in observation.keys()}
            if self.parallel
            else self.nextkey()
        )

        action, log_prob = self.explore_fn(self.state, keys, observation)

        return np.array(action), log_prob

    def train(
        self, env: GymEnv | EnvPoolEnv, n_env_steps: int, callbacks: list[Callback]
    ) -> None:
        return train(
            int(np.asarray(self.nextkey())[0]),
            self,
            env,
            n_env_steps,
            EnvType.SINGLE if self.config.env_cfg.n_agents == 1 else EnvType.PARALLEL,
            EnvProcs.ONE if self.config.env_cfg.n_envs == 1 else EnvProcs.MANY,
            AlgoType.ON_POLICY,
            saver=self.saver,
            callbacks=callbacks,
        )

    def resume(
        self, env: GymEnv | EnvPoolEnv, n_env_steps: int, callbacks: list[Callback]
    ) -> None:
        step = self.restore()

        return train(
            int(np.asarray(self.nextkey())[0]),
            self,
            env,
            n_env_steps,
            EnvType.SINGLE if self.config.env_cfg.n_agents == 1 else EnvType.PARALLEL,
            EnvProcs.ONE if self.config.env_cfg.n_envs == 1 else EnvProcs.MANY,
            AlgoType.ON_POLICY,
            start_step=step,
            saver=self.saver,
            callbacks=callbacks,
        )
