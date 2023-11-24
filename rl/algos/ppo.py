from typing import Callable, TypeVar

import chex
import distrax as dx
from flax import struct
import flax.linen as nn
from flax.training import train_state
import gymnasium as gym
from gymnasium import spaces
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from rl import Base, Params
from rl.buffer import OnPolicyBuffer, OnPolicyExp
from rl.loss import loss_policy_ppo_discrete, loss_value_clip
from rl.modules import modules_factory, create_params
from rl.timesteps import calculate_gaes_targets

EnvpoolEnv = TypeVar("EnvpoolEnv")


@chex.dataclass
class ParamsPPO:
    params_policy: Params
    params_value: Params
    params_encoder: Params


class TrainStatePPO(train_state.TrainState):
    policy_fn: Callable = struct.field(pytree_node=False)
    value_fn: Callable = struct.field(pytree_node=False)
    encoder_fn: Callable = struct.field(pytree_node=False)


def create_modules(
    observation_space: spaces.Space,
    action_space: spaces.Discrete,
    shared_encoder: bool,
    *,
    rearrange_pattern: str,
) -> tuple[nn.Module, nn.Module]:
    modules = modules_factory(
        observation_space,
        action_space,
        shared_encoder,
        rearrange_pattern=rearrange_pattern,
    )
    return tuple(modules.values())


def create_params_ppo(
    key: jax.Array,
    policy: nn.Module,
    value: nn.Module,
    encoder: nn.Module,
    observation_space: spaces.Space,
    *,
    shared_encoder: bool = False,
) -> ParamsPPO:
    key1, key2, key3 = jax.random.split(key, 3)
    if shared_encoder:
        if len(observation_space.shape) == 3:
            hidden_shape = (1, 512)
        else:
            hidden_shape = (1, 64)

        return ParamsPPO(
            params_policy=create_params(key1, policy, hidden_shape),
            params_value=create_params(key2, value, hidden_shape),
            params_encoder=create_params(key3, encoder, observation_space.shape),
        )
    else:
        return ParamsPPO(
            params_policy=create_params(key1, policy, observation_space.shape),
            params_value=create_params(key2, value, observation_space.shape),
            params_encoder=create_params(key3, encoder, observation_space.shape),
        )


def create_train_state(
    policy: nn.Module,
    value: nn.Module,
    encoder: nn.Module,
    params_ppo: ParamsPPO,
    config: ml_collections.ConfigDict,
    *,
    n_envs: int = 1,
) -> TrainStatePPO:
    num_batches = config.max_buffer_size // config.batch_size
    if config.learning_rate_annealing:
        n_updates = (
            config.n_env_steps
            * n_envs
            // config.max_buffer_size
            * config.num_epochs
            * num_batches
        )
        learning_rate = optax.linear_schedule(config.learning_rate, 0.0, n_updates, 0)
    else:
        learning_rate = config.learning_rate

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm), optax.adam(learning_rate)
    )

    return TrainStatePPO.create(
        apply_fn=None,
        params=params_ppo,
        tx=tx,
        policy_fn=policy.apply,
        value_fn=value.apply,
        encoder_fn=encoder.apply,
    )


def get_logits_and_log_probs(
    params_policy: Params,
    params_encoder: Params,
    policy_fn: Callable,
    encoder_fn: Callable,
    observations: jax.Array,
):
    hiddens = encoder_fn({"params": params_encoder}, observations)
    logits = policy_fn({"params": params_policy}, hiddens)
    return logits, nn.log_softmax(logits)


def compute_values(
    params: ParamsPPO,
    value_fn: Callable,
    encoder_fn: Callable,
    observations: jax.Array,
):
    hiddens = encoder_fn({"params": params.params_encoder}, observations)
    return value_fn({"params": params.params_value}, hiddens)


def loss_policy_fn(
    params_policy: Params,
    params_encoder: Params,
    policy_fn: Callable,
    encoder_fn: Callable,
    observations: jax.Array,
    actions: jax.Array,
    log_probs_old: jax.Array,
    gaes: jax.Array,
    clip_eps: float,
    entropy_coef: float,
):
    logits, all_log_probs = get_logits_and_log_probs(
        params_policy, params_encoder, policy_fn, encoder_fn, observations
    )
    log_probs = jnp.take_along_axis(all_log_probs, actions, axis=-1)

    return loss_policy_ppo_discrete(
        logits, log_probs, log_probs_old, gaes, clip_eps, entropy_coef
    )


def loss_value_fn(
    params_value: Params,
    params_encoder: Params,
    value_fn: Callable,
    encoder_fn: Callable,
    observations: jax.Array,
    targets: jax.Array,
    values_old: jax.Array,
    clip_eps: jax.Array,
):
    hiddens = encoder_fn({"params": params_encoder}, observations)
    values = value_fn({"params": params_value}, hiddens)
    return loss_value_clip(values, targets, values_old, clip_eps)


def loss_fn(
    params: ParamsPPO,
    policy_fn: Callable,
    value_fn: Callable,
    encoder_fn: Callable,
    batch: tuple[jax.Array],
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
):
    observations, actions, log_probs_old, gaes, targets, values_old = batch

    loss_policy, info_policy = loss_policy_fn(
        params.params_policy,
        params.params_encoder,
        policy_fn,
        encoder_fn,
        observations,
        actions,
        log_probs_old,
        gaes,
        clip_eps,
        entropy_coef,
    )

    loss_value, info_value = loss_value_fn(
        params.params_value,
        params.params_encoder,
        value_fn,
        encoder_fn,
        observations,
        targets,
        values_old,
        clip_eps,
    )

    infos = info_value | info_policy
    return loss_policy + value_coef * loss_value, infos


def explore(
    key: jax.Array,
    policy_fn: Callable,
    encoder_fn: Callable,
    params: ParamsPPO,
    observations: jax.Array,
) -> jax.Array:
    hiddens = encoder_fn({"params": params.params_encoder}, observations)
    logits = policy_fn({"params": params.params_policy}, hiddens)
    return dx.Categorical(logits=logits).sample_and_log_prob(seed=key)


def explore_unbatched(
    key: jax.Array,
    policy_fn: Callable,
    encoder_fn: Callable,
    params: ParamsPPO,
    observation: jax.Array,
) -> jax.Array:
    observations = jnp.expand_dims(observation, axis=0)
    actions, log_probs = explore(key, policy_fn, encoder_fn, params, observations)
    return jnp.squeeze(actions, axis=0), jnp.squeeze(log_probs, axis=0)


def process_experience(
    params: ParamsPPO,
    train_state: TrainStatePPO,
    gamma: float,
    _lambda: float,
    normalize: bool,
    sample: list[OnPolicyExp],
):
    from rl.buffer import array_of_name

    observations = array_of_name(sample, "observation")
    values = jax.jit(compute_values, static_argnums=(1, 2))(
        params, train_state.value_fn, train_state.encoder_fn, observations
    )

    next_observations = array_of_name(sample, "next_observation")
    next_values = jax.jit(compute_values, static_argnums=(1, 2))(
        params, train_state.value_fn, train_state.encoder_fn, next_observations
    )

    not_dones = 1.0 - array_of_name(sample, "done")[..., None]
    discounts = gamma * not_dones

    rewards = array_of_name(sample, "reward")[..., None]
    gaes, targets = jax.jit(calculate_gaes_targets, static_argnums=(4, 5))(
        values, next_values, discounts, rewards, _lambda, normalize
    )

    actions = array_of_name(sample, "action")[..., None]
    log_probs = array_of_name(sample, "log_prob")[..., None]
    return observations, actions, log_probs, gaes, targets, values


def process_experience_vectorized(
    params: ParamsPPO,
    train_state: TrainStatePPO,
    gamma: float,
    _lambda: float,
    normalize: bool,
    sample: list[OnPolicyExp],
):
    from rl.buffer import array_of_name

    observations = array_of_name(sample, "observation")  # T, E, size
    values = jax.vmap(
        jax.jit(compute_values, static_argnums=(1, 2)),
        in_axes=(None, None, None, 1),
        out_axes=1,
    )(params, train_state.value_fn, train_state.encoder_fn, observations)

    next_observations = array_of_name(sample, "next_observation")
    next_values = jax.vmap(
        jax.jit(compute_values, static_argnums=(1, 2)),
        in_axes=(None, None, None, 1),
        out_axes=1,
    )(params, train_state.value_fn, train_state.encoder_fn, next_observations)

    not_dones = 1.0 - array_of_name(sample, "done")[..., None]
    discounts = gamma * not_dones

    rewards = array_of_name(sample, "reward")[..., None]
    gaes, targets = jax.vmap(
        jax.jit(calculate_gaes_targets, static_argnums=(4, 5)),
        in_axes=(1, 1, 1, 1, None, None),
        out_axes=1,
    )(values, next_values, discounts, rewards, _lambda, normalize)

    actions = array_of_name(sample, "action")[..., None]
    log_probs = array_of_name(sample, "log_prob")[..., None]

    observations = jnp.reshape(observations, (-1, *observations.shape[2:]))
    actions = jnp.reshape(actions, (-1, *actions.shape[2:]))
    log_probs = jnp.reshape(log_probs, (-1, *log_probs.shape[2:]))
    gaes = jnp.reshape(gaes, (-1, *gaes.shape[2:]))
    targets = jnp.reshape(targets, (-1, *targets.shape[2:]))
    values = jnp.reshape(values, (-1, *values.shape[2:]))

    return observations, actions, log_probs, gaes, targets, values


def update_step(
    key: jax.Array,
    state: TrainStatePPO,
    experiences: tuple,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    batch_size: int,
):
    num_elems = experiences[0].shape[0]
    iterations = num_elems // batch_size
    inds = jax.random.permutation(key, num_elems)[: iterations * batch_size]

    experiences = jax.tree_util.tree_map(
        lambda x: x[inds].reshape((iterations, batch_size) + x.shape[1:]),
        experiences,
    )

    loss = 0.0
    for batch in zip(*experiences):
        (l, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params,
            policy_fn=state.policy_fn,
            value_fn=state.value_fn,
            encoder_fn=state.encoder_fn,
            batch=batch,
            clip_eps=clip_eps,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
        )
        loss += l
        state = state.apply_gradients(grads=grads)
    return state, loss, info


class PPO(Base):
    def __init__(
        self,
        seed: int,
        env: gym.Env,
        config: ml_collections.ConfigDict,
        *,
        create_modules: Callable[..., tuple[nn.Module, nn.Module]] = create_modules,
        create_params: Callable[..., ParamsPPO] = create_params_ppo,
        rearrange_pattern: str = "b h w c -> b h w c",
        n_envs: int = 1,
    ):
        Base.__init__(self, seed)
        self.config = config

        policy, value, encoder = create_modules(
            config.observation_space,
            config.action_space,
            config.shared_encoder,
            rearrange_pattern=rearrange_pattern,
        )
        params = create_params(
            self.nextkey(),
            policy,
            value,
            encoder,
            config.observation_space,
            shared_encoder=config.shared_encoder,
        )
        self.state = create_train_state(
            policy, value, encoder, params, self.config, n_envs=n_envs
        )

        self.explore_fn = explore if n_envs > 1 else explore_unbatched
        self.explore_fn = jax.jit(self.explore_fn, static_argnums=(1, 2))

        self.process_experience_fn = (
            process_experience_vectorized if n_envs > 1 else process_experience
        )

        self.n_envs = n_envs

    def select_action(self, observation: jax.Array) -> jax.Array:
        action, log_prob = self.explore_fn(
            self.nextkey(),
            self.state.policy_fn,
            self.state.encoder_fn,
            self.state.params,
            observation,
        )
        return action, log_prob

    def explore(self, observation: jax.Array) -> jax.Array:
        action, log_prob = self.explore_fn(
            self.nextkey(),
            self.state.policy_fn,
            self.state.encoder_fn,
            self.state.params,
            observation,
        )
        return action, log_prob

    def update(self, buffer: OnPolicyBuffer):
        sample = buffer.sample()
        experiences = self.process_experience_fn(
            self.state.params,
            self.state,
            self.config.gamma,
            self.config._lambda,
            self.config.normalize,
            sample,
        )

        loss = 0.0
        for epoch in range(self.config.num_epochs):
            self.state, l, info = jax.jit(update_step, static_argnums=(3, 4, 5, 6))(
                self.nextkey(),
                self.state,
                experiences,
                self.config.clip_eps,
                self.config.entropy_coef,
                self.config.value_coef,
                self.config.batch_size,
            )
            loss += l

        loss /= self.config.num_epochs
        return info


def train(seed: int, ppo: PPO, env: gym.Env, n_env_steps: int):
    assert ppo.n_envs == 1

    buffer = OnPolicyBuffer(seed, ppo.config.max_buffer_size)

    observation, info = env.reset(seed=seed + 1)
    episode_return = 0.0

    update_info = {"kl_divergence": 0.0}

    for step in range(1, n_env_steps + 1):
        action, log_prob = ppo.explore(np.array(observation))
        next_observation, reward, done, trunc, info = env.step(int(action))
        episode_return += reward

        buffer.add(
            OnPolicyExp(
                observation=observation,
                action=action,
                reward=reward,
                done=done,
                next_observation=next_observation,
                log_prob=log_prob,
            )
        )

        if done or trunc:
            print(step, " > ", episode_return, " | ", update_info["kl_divergence"])
            episode_return = 0.0
            next_observation, info = env.reset()

        if len(buffer) >= ppo.config.max_buffer_size:
            update_info = ppo.update(buffer)

        observation = next_observation


def train_vectorized(seed: int, ppo: PPO, env: EnvpoolEnv, n_env_steps: int):
    assert ppo.n_envs > 1

    buffer = OnPolicyBuffer(seed, ppo.config.max_buffer_size)

    observation, info = env.reset()
    episode_return = np.zeros((observation.shape[0],))

    update_info = {"kl_divergence": 0.0}

    for step in range(1, n_env_steps + 1):
        action, log_prob = ppo.explore(observation)
        next_observation, reward, done, trunc, info = env.step(np.array(action))
        episode_return += reward

        buffer.add(
            OnPolicyExp(
                observation=observation,
                action=action,
                reward=reward,
                done=done,
                next_observation=next_observation,
                log_prob=log_prob,
            )
        )

        for i, (d, t) in enumerate(zip(done, trunc)):
            if d or t:
                if i == 0:
                    print(
                        step,
                        " > ",
                        episode_return[i],
                        " | ",
                        update_info["kl_divergence"],
                    )
                episode_return[i] = 0.0

        if len(buffer) >= ppo.config.max_buffer_size:
            update_info = ppo.update(buffer)

        observation = next_observation
