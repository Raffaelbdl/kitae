from dataclasses import dataclass
from typing import Callable, Type, Iterable

import chex
from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
from gymnasium import spaces
import jax
import jax.numpy as jnp
import ml_collections
import optax

from rl.config import AlgoConfig
from rl.modules.modules import init_params, PassThrough
from rl.modules.encoder import encoder_factory
from rl.modules.optimizer import linear_learning_rate_schedule
from rl.modules.policy import PolicyNormalExternalStd
from rl.types import Params


class QValueDiscreteOutput(nn.Module):
    num_outputs: int

    @nn.compact
    def __call__(self, x: jax.Array):
        return nn.Dense(
            features=self.num_outputs,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.constant(0.0),
        )(x)


class QValueContinousOutput(nn.Module):
    observation_space: spaces.Space

    @nn.compact
    def __call__(self, x: jax.Array, a: jax.Array):
        x = jnp.concatenate([x, a], axis=-1)
        x = nn.relu(nn.Dense(64)(x))
        return nn.Dense(
            features=1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.constant(0.0),
        )(x)


class DoubleQValueContinuousOutput(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array, a: jax.Array):
        return QValueContinousOutput()(x, a), QValueContinousOutput()(x, a)


@chex.dataclass
class ParamsPolicyQValue:
    params_encoder: Params
    params_policy: Params
    params_qvalue: Params


class TrainStatePolicyQvalue(TrainState):
    encoder_fn: Callable = struct.field(pytree_node=False)
    policy_fn: Callable = struct.field(pytree_node=False)
    qvalue_fn: Callable = struct.field(pytree_node=False)

    params_target: ParamsPolicyQValue = struct.field(pytree_node=True)


@dataclass
class QModules:
    encoder: nn.Module
    qvalue: nn.Module
    policy: nn.Module = None


def create_q_modules(
    observation_space: spaces.Space,
    action_space: spaces.Discrete,
    shared_encoder: bool,
    *,
    rearrange_pattern: str = "b h w c -> b h w c",
    preprocess_fn: Callable = None,
) -> QModules:
    encoder = encoder_factory(
        observation_space,
        rearrange_pattern=rearrange_pattern,
        preprocess_fn=preprocess_fn,
    )

    if shared_encoder:
        return QModules(encoder=encoder, qvalue=QValueDiscreteOutput(action_space.n))

    class QValue(nn.Module):
        @nn.compact
        def __call__(self, x: jax.Array):
            x = encoder()(x)
            return QValueDiscreteOutput(action_space.n)(x)

    return QModules(encoder=PassThrough(), qvalue=QValue())


def create_train_state_qvalue(
    qvalue: nn.Module,
    params: Params,
    config: ml_collections.ConfigDict,
    *,
    n_envs: int = 1,
) -> TrainState:
    learning_rate = config.update_cfg.learning_rate

    if config.update_cfg.learning_rate_annealing:
        learning_rate = linear_learning_rate_schedule(
            learning_rate,
            0.0,
            n_envs=n_envs,
            n_env_steps=config.train_cfg.n_env_steps,
            max_buffer_size=config.update_cfg.max_buffer_size,
            batch_size=config.update_cfg.batch_size,
            num_epochs=config.update_cfg.num_epochs,
        )

    tx = optax.chain(
        optax.clip_by_global_norm(config.update_cfg.max_grad_norm),
        optax.adam(learning_rate, eps=1e-5),
    )

    return TrainState.create(
        apply_fn=qvalue.apply,
        params=params,
        tx=tx,
    )


def train_state_qvalue_factory(
    key: jax.Array,
    config: AlgoConfig,
    *,
    rearrange_pattern: str,
    preprocess_fn: Callable,
    tabulate: bool = False,
) -> TrainState:
    qvalue = create_q_modules(
        config.env_cfg.observation_space,
        config.env_cfg.action_space,
        False,
        rearrange_pattern=rearrange_pattern,
        preprocess_fn=preprocess_fn,
    ).qvalue
    params = init_params(
        key, qvalue, config.env_cfg.observation_space.shape, tabulate=tabulate
    )
    state = create_train_state_qvalue(
        qvalue, params, config, n_envs=config.env_cfg.n_envs
    )
    return state


def create_policy_q_modules(
    observation_space: spaces.Space,
    action_space: spaces.Box,
    shared_encoder: bool,
    *,
    rearrange_pattern: str = "b h w c -> b h w c",
    preprocess_fn: Callable = None,
) -> QModules:
    encoder = encoder_factory(
        observation_space,
        rearrange_pattern=rearrange_pattern,
        preprocess_fn=preprocess_fn,
    )

    if shared_encoder:
        return QModules(
            encoder=encoder(),
            qvalue=DoubleQValueContinuousOutput(),
            policy=PolicyNormalExternalStd(
                action_space.shape[-1],
                (action_space.high - action_space.low) / 2.0,
                (action_space.high + action_space.low) / 2.0,
            ),
        )

    class Policy(nn.Module):
        @nn.compact
        def __call__(self, x: jax.Array):
            x = encoder()(x)
            return PolicyNormalExternalStd(
                action_space.shape[-1],
                (action_space.high - action_space.low) / 2.0,
                (action_space.high + action_space.low) / 2.0,
            )(x)

    class QValue(nn.Module):
        @nn.compact
        def __call__(self, x: jax.Array, a: jax.Array):
            x = encoder()(x)
            return DoubleQValueContinuousOutput()(x, a)

    return QModules(encoder=PassThrough(), qvalue=QValue(), policy=Policy())


def create_params_policy_qvalue(
    key: jax.Array,
    modules: QModules,
    observation_space: spaces.Space,
    action_space: spaces.Box,
    *,
    shared_encoder: bool = False,
    tabulate: bool = False,
) -> ParamsPolicyQValue:
    key1, key2, key3 = jax.random.split(key, 3)
    if shared_encoder:
        if len(observation_space.shape) == 3:
            hidden_shape = (512,)
        else:
            hidden_shape = (64,)

        return ParamsPolicyQValue(
            params_encoder=init_params(
                key1, modules.encoder, observation_space.shape, tabulate=tabulate
            ),
            params_policy=init_params(
                key2, modules.policy, hidden_shape, tabulate=tabulate
            ),
            params_qvalue=init_params(
                key3,
                modules.qvalue,
                [hidden_shape, action_space.shape],
                tabulate=tabulate,
            ),
        )

    return ParamsPolicyQValue(
        params_encoder=init_params(
            key1, modules.encoder, observation_space.shape, tabulate=tabulate
        ),
        params_policy=init_params(
            key2, modules.policy, observation_space.shape, tabulate=tabulate
        ),
        params_qvalue=init_params(
            key3,
            modules.qvalue,
            [observation_space.shape, action_space.shape],
            tabulate=tabulate,
        ),
    )


def create_train_state_policy_qvalue(
    modules: QModules,
    params: ParamsPolicyQValue,
    config: AlgoConfig,
    *,
    n_envs: int = 1,
) -> TrainStatePolicyQvalue:
    learning_rate = config.update_cfg.learning_rate

    if config.update_cfg.learning_rate_annealing:
        learning_rate = linear_learning_rate_schedule(
            learning_rate,
            0.0,
            n_envs=n_envs,
            n_env_steps=config.train_cfg.n_env_steps,
            max_buffer_size=config.update_cfg.max_buffer_size,
            batch_size=config.update_cfg.batch_size,
            num_epochs=config.update_cfg.num_epochs,
        )

    tx = optax.chain(
        optax.clip_by_global_norm(config.update_cfg.max_grad_norm),
        optax.adam(learning_rate, eps=1e-5),
    )

    return TrainStatePolicyQvalue.create(
        apply_fn=None,
        params=params,
        tx=tx,
        encoder_fn=modules.encoder.apply,
        qvalue_fn=modules.qvalue.apply,
        policy_fn=modules.policy.apply,
        params_target=params,
    )


def train_state_policy_qvalue_factory(
    key: jax.Array,
    config: AlgoConfig,
    *,
    rearrange_pattern: str,
    preprocess_fn: Callable,
    tabulate: bool = False,
) -> TrainStatePolicyQvalue:
    modules = create_policy_q_modules(
        config.env_cfg.observation_space,
        config.env_cfg.action_space,
        False,
        rearrange_pattern=rearrange_pattern,
        preprocess_fn=preprocess_fn,
    )
    params = create_params_policy_qvalue(
        key,
        modules,
        config.env_cfg.observation_space,
        config.env_cfg.action_space,
        shared_encoder=False,
        tabulate=tabulate,
    )
    state = create_train_state_policy_qvalue(
        modules, params, config, n_envs=config.env_cfg.n_envs * config.env_cfg.n_agents
    )
    return state


def qvalue_factory(
    observation_space: spaces.Space,
    action_space: spaces.Space,
    *,
    rearrange_pattern: str = "b h w c -> b h w c",
    preprocess_fn: Callable = None,
) -> Type[nn.Module]:
    encoder = encoder_factory(
        observation_space,
        rearrange_pattern=rearrange_pattern,
        preprocess_fn=preprocess_fn,
    )

    if isinstance(action_space, spaces.Discrete):

        class QValue(nn.Module):
            def setup(self) -> None:
                self.encoder = encoder()

            @nn.compact
            def __call__(self, x: jax.Array) -> jax.Array:
                x = self.encoder(x)
                return nn.Dense(action_space.n)(x)

    elif isinstance(action_space, spaces.Box):

        class QValue(nn.Module):
            def setup(self) -> None:
                self.encoder = encoder()

            @nn.compact
            def __call__(self, x: jax.Array, a: jax.Array) -> jax.Array:
                if len(observation_space.shape) == 1:
                    x = self.encoder(jnp.concatenate([x, a], axis=-1))

                elif len(observation_space.shape) == 3:
                    x = self.encoder(x)
                    x = jnp.concatenate([x, a], axis=-1)

                else:
                    raise NotImplementedError

                return nn.Dense(1)(x)

    else:
        raise NotImplementedError

    return QValue


def make_double_q_value(q1: nn.Module, q2: nn.Module) -> nn.Module:
    class DoubleQValue(nn.Module):
        def setup(self) -> None:
            self.q1 = q1
            self.q2 = q2

        def __call__(self, *arrays: Iterable[jax.Array]) -> tuple[jax.Array, jax.Array]:
            return self.q1(*arrays), self.q2(*arrays)

    return DoubleQValue()
