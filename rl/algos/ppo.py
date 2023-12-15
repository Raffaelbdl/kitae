from dataclasses import dataclass
from typing import Callable

import distrax as dx
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from rl.base import Base, EnvType, EnvProcs, AlgoType
from rl.buffer import OnPolicyBuffer, OnPolicyExp
from rl.loss import loss_policy_ppo, loss_value_clip
from rl.timesteps import calculate_gaes_targets
from rl.train import train

from rl.types import GymEnv, EnvPoolEnv

from rl.modules.policy_value import train_state_policy_value_factory
from rl.modules.policy_value import TrainStatePolicyValue, ParamsPolicyValue

from rl.algos.general_fns import fn_parallel
from rl.buffer import stack_experiences

from rl.config import AlgoConfig, AlgoParams
import distrax as dx

from absl import logging
from rl.distribution import get_log_probs


@dataclass
class PPOParams(AlgoParams):
    gamma: float
    _lambda: float
    clip_eps: float
    entropy_coef: float
    value_coef: float
    normalize: bool


def loss_factory(
    train_state: TrainStatePolicyValue,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
) -> Callable:
    @jax.jit
    def fn(params: ParamsPolicyValue, batch: tuple[jax.Array]):
        observations, actions, log_probs_old, gaes, targets, values_old = batch
        hiddens = train_state.encoder_fn(
            {"params": params.params_encoder}, observations
        )

        dists: dx.Categorical = train_state.policy_fn(
            {"params": params.params_policy}, hiddens
        )
        log_probs, log_probs_old = get_log_probs(dists, actions, log_probs_old)

        loss_policy, info_policy = loss_policy_ppo(
            dists, log_probs, log_probs_old, gaes, clip_eps, entropy_coef
        )

        values = train_state.value_fn({"params": params.params_value}, hiddens)
        loss_value, info_value = loss_value_clip(values, targets, values_old, clip_eps)

        loss = loss_policy + value_coef * loss_value
        info = info_policy | info_value
        info["total_loss"] = loss

        return loss, info

    return fn


def explore_factory(
    train_state: TrainStatePolicyValue,
    algo_params: PPOParams,
) -> Callable:
    @jax.jit
    def fn(
        params: ParamsPolicyValue, key: jax.Array, observations: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        hiddens = train_state.encoder_fn(
            {"params": params.params_encoder}, observations
        )
        dists = train_state.policy_fn({"params": params.params_policy}, hiddens)
        outputs = dists.sample_and_log_prob(seed=key)

        return outputs

    return fn


def process_experience_factory(
    train_state: TrainStatePolicyValue,
    algo_params: PPOParams,
    vectorized: bool,
    parallel: bool,
):
    def compute_values_gaes(
        params: ParamsPolicyValue,
        observations: jax.Array,
        next_observations: jax.Array,
        dones: jax.Array,
        rewards: jax.Array,
    ):
        all_obs = jnp.concatenate([observations, next_observations[-1:]], axis=0)
        all_hiddens = train_state.encoder_fn({"params": params.params_encoder}, all_obs)
        all_values = train_state.value_fn({"params": params.params_value}, all_hiddens)

        values = all_values[:-1]
        next_values = all_values[1:]

        not_dones = 1.0 - dones[..., None]
        discounts = algo_params.gamma * not_dones

        rewards = rewards[..., None]
        gaes, targets = calculate_gaes_targets(
            values,
            next_values,
            discounts,
            rewards,
            algo_params._lambda,
            algo_params.normalize,
        )

        return gaes, targets, values

    gaes_fn = compute_values_gaes
    if vectorized:
        gaes_fn = jax.vmap(gaes_fn, in_axes=(None, 1, 1, 1, 1), out_axes=1)
    if parallel:
        gaes_fn = fn_parallel(gaes_fn)

    @jax.jit
    def fn(params: ParamsPolicyValue, sample: list[OnPolicyExp]):
        stacked = stack_experiences(sample)

        observations = stacked.observation
        gaes, targets, values = gaes_fn(
            params, observations, stacked.next_observation, stacked.done, stacked.reward
        )

        actions = stacked.action
        log_probs = stacked.log_prob

        logging.debug(
            (
                "Experiences shapes : \n"
                + f"observations : {observations.shape} \n"
                + f"actions : {actions.shape} \n"
                + f"log_probs : {log_probs.shape} \n"
                + f"gaes : {gaes.shape}"
            )
        )
        return observations, actions, log_probs, gaes, targets, values

    return fn


def update_step_factory(
    train_state: TrainStatePolicyValue, config: ml_collections.ConfigDict
):
    loss_fn = loss_factory(
        train_state,
        config.algo_params.clip_eps,
        config.algo_params.entropy_coef,
        config.algo_params.value_coef,
    )

    @jax.jit
    def fn(state: TrainStatePolicyValue, key: jax.Array, experiences: tuple[jax.Array]):
        num_elems = experiences[0].shape[0]
        iterations = num_elems // config.update_cfg.batch_size
        inds = jax.random.permutation(key, num_elems)[
            : iterations * config.update_cfg.batch_size
        ]

        experiences = jax.tree_util.tree_map(
            lambda x: x[inds].reshape(
                (iterations, config.update_cfg.batch_size) + x.shape[1:]
            ),
            experiences,
        )

        loss = 0.0
        for batch in zip(*experiences):
            (l, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params, batch=batch
            )
            loss += l

            state = state.apply_gradients(grads=grads)
        return state, loss, info

    return fn


class PPO(Base):
    def __init__(
        self,
        config: AlgoConfig,
        *,
        rearrange_pattern: str = "b h w c -> b h w c",
        preprocess_fn: Callable = None,
        run_name: str = None,
        tabulate: bool = False,
    ):
        Base.__init__(
            self,
            config=config,
            train_state_factory=train_state_policy_value_factory,
            explore_factory=explore_factory,
            process_experience_factory=process_experience_factory,
            update_step_factory=update_step_factory,
            rearrange_pattern=rearrange_pattern,
            preprocess_fn=preprocess_fn,
            run_name=run_name,
            tabulate=tabulate,
        )

    def select_action(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        return self.explore(observation)

    def explore(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        keys = (
            {a: self.nextkey() for a in observation.keys()}
            if self.parallel
            else self.nextkey()
        )

        action, log_prob = self.explore_fn(self.state.params, keys, observation)

        return action, log_prob

    def should_update(self, step: int, buffer: OnPolicyBuffer) -> None:
        return len(buffer) >= self.config.update_cfg.max_buffer_size

    def update(self, buffer: OnPolicyBuffer):
        def fn(state: TrainStatePolicyValue, key: jax.Array, sample: tuple):
            experiences = self.process_experience_fn(state.params, sample)

            loss = 0.0
            for epoch in range(self.config.update_cfg.n_epochs):
                key, _k = jax.random.split(key)
                state, l, info = self.update_step_fn(state, _k, experiences)
                loss += l

            loss /= self.config.update_cfg.n_epochs
            info["total_loss"] = loss
            return state, info

        sample = buffer.sample(self.config.update_cfg.batch_size)
        self.state, info = fn(self.state, self.nextkey(), sample)
        return info

    def train(self, env: GymEnv | EnvPoolEnv, n_env_steps: int, callbacks: list):
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

    def resume(self, env: GymEnv | EnvPoolEnv, n_env_steps: int, callbacks: list):
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
