from typing import Callable

import jax
from jax import numpy as jnp
import ml_collections
import numpy as np

from rl.base import Base, EnvType, EnvProcs, AlgoType
from rl.buffer import OnPolicyBuffer
from rl.loss import loss_shannon_jensen_divergence
from rl.train import train_population

from rl.types import GymEnv, EnvPoolEnv

from rl.algos.ppo import loss_factory as loss_single_factory
from rl.algos.ppo import explore_factory
from rl.algos.ppo import process_experience_factory
from rl.modules.policy_value import TrainStatePolicyValue, ParamsPolicyValue
from rl.modules.policy_value import train_state_policy_value_population_factory


def loss_factory(
    train_state: TrainStatePolicyValue,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    jsd_coef: float,
) -> Callable:
    loss_single_fn = loss_single_factory(
        train_state, clip_eps, entropy_coef, value_coef
    )

    @jax.jit
    def fn(
        params_population: list[ParamsPolicyValue], batch: list[tuple[jax.Array]]
    ) -> float:
        loss, infos = 0.0, {}
        logits_pop = []
        entropy_pop = []

        for agent_id in range(len(params_population)):
            l, i = loss_single_fn(params_population[agent_id], batch[agent_id])
            loss += l
            logits_pop.append(i.pop("logits"))
            entropy_pop.append(i.pop("entropy"))
            infos[f"agent_{agent_id}"] = i

        if jsd_coef <= 0.0:
            return loss, infos

        logits_average = jnp.array(logits_pop).mean(axis=0)
        entropy_average = jnp.array(entropy_pop).mean(axis=0)[..., None]

        loss += jsd_coef * loss_shannon_jensen_divergence(
            logits_average, entropy_average
        )
        infos["population"] = {
            "loss_jsd": loss,
            "mean_population_entropy": jnp.mean(entropy_average),
        }
        return loss, infos

    return fn


def update_step_factory(
    train_state: TrainStatePolicyValue, config: ml_collections.ConfigDict
):
    loss_fn = loss_factory(
        train_state,
        config.clip_eps,
        config.entropy_coef,
        config.value_coef,
        config.jsd_coef,
    )

    @jax.jit
    def fn(
        state: TrainStatePolicyValue,
        key: jax.Array,
        experiences: list[tuple[jax.Array]],
    ):
        num_elems = experiences[0][0].shape[0]
        iterations = num_elems // config.batch_size
        inds = jax.random.permutation(key, num_elems)[: iterations * config.batch_size]

        experiences = jax.tree_util.tree_map(
            lambda x: x[inds].reshape((iterations, config.batch_size) + x.shape[1:]),
            experiences,
        )

        loss = 0.0
        for i in range(iterations):
            batch = [tuple(v[i] for v in e) for e in experiences]

            (l, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params, batch=batch
            )
            loss += l

            state = state.apply_gradients(grads=grads)
        return state, loss, info

    return fn


class PopulationPPO(Base):
    def __init__(
        self,
        seed: int,
        config: ml_collections.ConfigDict,
        *,
        rearrange_pattern: str = "b h w c -> b h w c",
        preprocess_fn: Callable = None,
        run_name: str = None,
        tabulate: bool = False,
    ):
        Base.__init__(
            self,
            seed=seed,
            config=config,
            train_state_factory=train_state_policy_value_population_factory,
            explore_factory=explore_factory,
            process_experience_factory=process_experience_factory,
            update_step_factory=update_step_factory,
            rearrange_pattern=rearrange_pattern,
            preprocess_fn=preprocess_fn,
            run_name=run_name,
            tabulate=tabulate,
        )

    def select_action(self, observations):
        return self.explore(observations)

    def explore(self, observations):
        actions, log_probs = [], []
        for i, obs in enumerate(observations):
            keys = (
                {a: self.nextkey() for a in obs.keys()}
                if self.parallel
                else self.nextkey()
            )
            action, log_prob = self.explore_fn(self.state.params[i], keys, obs)
            actions.append(action)
            log_probs.append(log_prob)

        return actions, log_probs

    def should_update(self, step: int, buffer: OnPolicyBuffer) -> None:
        return len(buffer) >= self.config.max_buffer_size

    def update(self, buffers: list[OnPolicyBuffer]):
        def fn(state: TrainStatePolicyValue, key: jax.Array, samples: list[tuple]):
            experiences = [
                self.process_experience_fn(state.params[i], samples[i])
                for i in range(len(samples))
            ]

            loss = 0.0
            for epoch in range(self.config.num_epochs):
                key, _k = jax.random.split(key)
                state, l, info = self.update_step_fn(state, _k, experiences)
                loss += l

            loss /= self.config.num_epochs
            info["total_loss"] = loss
            return state, info

        samples = [b.sample() for b in buffers]
        self.state, info = fn(self.state, self.nextkey(), samples)
        return info

    def train(self, env: list[GymEnv | EnvPoolEnv], n_env_steps: int, callbacks: list):
        return train_population(
            int(np.asarray(self.nextkey())[0]),
            self,
            env,
            n_env_steps,
            EnvType.SINGLE
            if self.config.env_config.n_agents == 1
            else EnvType.PARALLEL,
            EnvProcs.ONE if self.config.env_config.n_envs == 1 else EnvProcs.MANY,
            AlgoType.ON_POLICY,
            saver=self.saver,
            callbacks=callbacks,
        )

    def resume(self, env: list[GymEnv | EnvPoolEnv], n_env_steps: int, callbacks: list):
        step, self.state = self.saver.restore_latest_step(self.state)

        return train_population(
            int(np.asarray(self.nextkey())[0]),
            self,
            env,
            n_env_steps,
            EnvType.SINGLE
            if self.config.env_config.n_agents == 1
            else EnvType.PARALLEL,
            EnvProcs.ONE if self.config.env_config.n_envs == 1 else EnvProcs.MANY,
            AlgoType.ON_POLICY,
            start_step=step,
            saver=self.saver,
            callbacks=callbacks,
        )
