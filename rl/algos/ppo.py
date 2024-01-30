"""Proximal Policy Optimization (PPO)"""

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from rl.base import Base, EnvType, EnvProcs, AlgoType
from rl.buffer import OnPolicyBuffer, Experience
from rl.callbacks.callback import Callback
from rl.config import AlgoConfig, AlgoParams
from rl.types import GymEnv, EnvPoolEnv

from rl.distribution import get_log_probs
from rl.loss import loss_policy_ppo, loss_value_clip
from rl.modules.policy_value import (
    train_state_policy_value_factory,
    ParamsPolicyValue,
)
from rl.timesteps import calculate_gaes_targets
from rl.train import train


from rl.modules.train_state import PolicyValueTrainState


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


def explore_factory(
    train_state: PolicyValueTrainState, algo_params: PPOParams
) -> Callable:
    encoder_apply = train_state.encoder_fn
    policy_apply = train_state.policy_fn

    @jax.jit
    def fn(
        params: ParamsPolicyValue, key: jax.Array, observations: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        hiddens = encoder_apply({"params": params.params_encoder}, observations)
        dists = policy_apply({"params": params.params_policy}, *hiddens)
        outputs = dists.sample_and_log_prob(seed=key)

        return outputs

    return fn


def process_experience_factory(
    train_state: PolicyValueTrainState, algo_params: PPOParams
):
    encoder_apply = train_state.encoder_fn
    value_apply = train_state.value_fn

    @jax.jit
    def fn(ppo_state: PolicyValueTrainState, key: jax.Array, experience: Experience):
        all_obs = jnp.concatenate(
            [experience.observation, experience.next_observation[-1:]], axis=0
        )
        all_hiddens = encoder_apply(
            {"params": ppo_state.params.params_encoder}, all_obs
        )
        all_values = value_apply(
            {"params": ppo_state.params.params_value}, *all_hiddens
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


def update_step_factory(
    train_state: PolicyValueTrainState, config: AlgoConfig
) -> Callable:
    encoder_apply = train_state.encoder_fn
    policy_apply = train_state.policy_fn
    value_apply = train_state.value_fn

    def loss_fn(
        params: ParamsPolicyValue, batch: tuple[jax.Array]
    ) -> tuple[float, dict]:
        observations, actions, log_probs_old, gaes, targets, values_old = batch
        hiddens = encoder_apply({"params": params.params_encoder}, observations)

        dists = policy_apply({"params": params.params_policy}, *hiddens)
        log_probs, log_probs_old = get_log_probs(dists, actions, log_probs_old)
        loss_policy, info_policy = loss_policy_ppo(
            dists,
            log_probs,
            log_probs_old,
            gaes,
            config.algo_params.clip_eps,
            config.algo_params.entropy_coef,
        )

        values = value_apply({"params": params.params_value}, *hiddens)
        loss_value, info_value = loss_value_clip(
            values, targets, values_old, config.algo_params.clip_eps
        )

        loss = loss_policy + config.algo_params.value_coef * loss_value
        info = info_policy | info_value
        info["total_loss"] = loss

        return loss, info

    @jax.jit
    def fn(state: PolicyValueTrainState, key: jax.Array, experiences: tuple[jax.Array]):
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

    def should_update(self, step: int, buffer: OnPolicyBuffer) -> bool:
        return len(buffer) >= self.config.update_cfg.max_buffer_size

    def update(self, buffer: OnPolicyBuffer) -> dict:
        def fn(state: PolicyValueTrainState, key: jax.Array, sample: tuple):
            experiences = self.process_experience_fn(state, key, sample)

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
