from dataclasses import dataclass
from typing import Any, Callable

import chex
import distrax as dx
from flax import struct
from flax import linen as nn
from flax.training import train_state
from gymnasium import spaces
import jax
import jax.numpy as jnp
import ml_collections
import optax

from rl.config import AlgoConfig
from rl.modules.modules import init_params, PassThrough
from rl.modules.encoder import encoder_factory
from rl.modules.optimizer import linear_learning_rate_schedule

from rl.types import Params
from dx_tabulate import add_representer
from rl.modules.policy import Policy, PolicyCategorical, PolicyNormal


def policy_output_factory(action_space: spaces.Discrete) -> type[Policy]:
    if isinstance(action_space, spaces.Discrete):
        add_representer(dx.Categorical)
        return PolicyCategorical
    elif isinstance(action_space, spaces.Box):
        add_representer(dx.Normal)
        return PolicyNormal
    else:
        raise NotImplementedError


class ValueOutput(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array):
        return nn.Dense(
            features=1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.constant(0.0),
        )(x)


@dataclass
class PolicyValueModules:
    encoder: nn.Module
    policy: nn.Module
    value: nn.Module


@chex.dataclass
class ParamsPolicyValue:
    params_encoder: Params
    params_policy: Params
    params_value: Params


class TrainStatePolicyValue(train_state.TrainState):
    """Custom TrainState with additional apply_fns.

    Contrary to off-policy algorithms, on-policy algorithms often require
    shared encoders and simultaneous updates of policy and value modules.
    This class extends TrainState by storing all parameters in a single
    PyTree and by providing additional functions for each module call.

    Attributes:
        encoder_fn: A Callable for the encoder module.
            If the encoder is not shared, it is usually the `PassThrough` module.
        policy_fn: A Callable for the policy module.
        value_fn: A Callable for the value module.
    """

    encoder_fn: Callable = struct.field(pytree_node=False)
    policy_fn: Callable = struct.field(pytree_node=False)
    value_fn: Callable = struct.field(pytree_node=False)


def create_policy_value_modules(
    observation_space: spaces.Space,
    action_space: spaces.Discrete,
    shared_encoder: bool,
    *,
    rearrange_pattern: str,
    preprocess_fn: Callable,
):
    encoder = encoder_factory(
        observation_space,
        rearrange_pattern=rearrange_pattern,
        preprocess_fn=preprocess_fn,
    )

    policy_output = policy_output_factory(action_space)
    num_actions = (
        action_space.n
        if isinstance(action_space, spaces.Discrete)
        else action_space.shape[-1]
    )

    if shared_encoder:
        return PolicyValueModules(
            encoder=encoder(), policy=policy_output(num_actions), value=ValueOutput()
        )

    class Policy(nn.Module):
        @nn.compact
        def __call__(self, x: jax.Array):
            x = encoder()(x)
            return policy_output(num_actions)(x)

    class Value(nn.Module):
        @nn.compact
        def __call__(self, x: jax.Array):
            x = encoder()(x)
            return ValueOutput()(x)

    return PolicyValueModules(encoder=PassThrough(), policy=Policy(), value=Value())


def create_params_policy_value(
    key: jax.Array,
    modules: PolicyValueModules,
    observation_space: spaces.Space,
    *,
    shared_encoder: bool = False,
    tabulate: bool = False,
) -> ParamsPolicyValue:
    key1, key2, key3 = jax.random.split(key, 3)
    if shared_encoder:
        if len(observation_space.shape) == 3:
            hidden_shape = (512,)
        else:
            hidden_shape = (64,)

        return ParamsPolicyValue(
            params_encoder=init_params(
                key1, modules.encoder, observation_space.shape, tabulate=tabulate
            ),
            params_policy=init_params(
                key2, modules.policy, hidden_shape, tabulate=tabulate
            ),
            params_value=init_params(
                key3, modules.value, hidden_shape, tabulate=tabulate
            ),
        )
    return ParamsPolicyValue(
        params_encoder=init_params(
            key1, modules.encoder, observation_space.shape, tabulate=tabulate
        ),
        params_policy=init_params(
            key2, modules.policy, observation_space.shape, tabulate=tabulate
        ),
        params_value=init_params(
            key3, modules.value, observation_space.shape, tabulate=tabulate
        ),
    )


def create_train_state_policy_value(
    modules: PolicyValueModules,
    params: ParamsPolicyValue,
    config: ml_collections.ConfigDict,
    *,
    n_envs: int = 1,
) -> TrainStatePolicyValue:
    learning_rate = config.update_cfg.learning_rate

    if config.update_cfg.learning_rate_annealing:
        learning_rate = linear_learning_rate_schedule(
            learning_rate,
            0.0,
            n_envs=n_envs,
            n_env_steps=config.train_cfg.n_env_steps,
            max_buffer_size=config.update_cfg.max_buffer_size,
            batch_size=config.update_cfg.batch_size,
            num_epochs=config.update_cfg.n_epochs,
        )

    tx = optax.chain(
        optax.clip_by_global_norm(config.update_cfg.max_grad_norm),
        optax.adam(learning_rate, eps=1e-5),
    )

    return TrainStatePolicyValue.create(
        apply_fn=None,
        params=params,
        tx=tx,
        policy_fn=modules.policy.apply,
        value_fn=modules.value.apply,
        encoder_fn=modules.encoder.apply,
    )


def train_state_policy_value_factory(
    key: jax.Array,
    config: ml_collections.ConfigDict,
    *,
    rearrange_pattern: str,
    preprocess_fn: Callable,
    tabulate: bool = False,
) -> TrainStatePolicyValue:
    modules = create_policy_value_modules(
        config.env_cfg.observation_space,
        config.env_cfg.action_space,
        config.update_cfg.shared_encoder,
        rearrange_pattern=rearrange_pattern,
        preprocess_fn=preprocess_fn,
    )
    params = create_params_policy_value(
        key,
        modules,
        config.env_cfg.observation_space,
        shared_encoder=config.update_cfg.shared_encoder,
        tabulate=tabulate,
    )
    state = create_train_state_policy_value(
        modules,
        params,
        config,
        n_envs=config.env_cfg.n_envs * config.env_cfg.n_agents,
    )
    return state


def train_state_policy_value_population_factory(
    key: jax.Array,
    config: ml_collections,
    *,
    rearrange_pattern: str,
    preprocess_fn: Callable,
    tabulate: bool = False,
) -> TrainStatePolicyValue:
    modules = create_policy_value_modules(
        config.env_config.observation_space,
        config.env_config.action_space,
        config.shared_encoder,
        rearrange_pattern=rearrange_pattern,
        preprocess_fn=preprocess_fn,
    )
    params_population = [
        create_params_policy_value(
            key,
            modules,
            config.env_config.observation_space,
            shared_encoder=config.shared_encoder,
            tabulate=(i == 0 and tabulate),
        )
        for i in range(config.population_size)
    ]
    state = create_train_state_policy_value(
        modules,
        params_population,
        config,
        n_envs=config.env_config.n_envs * config.env_config.n_agents,
    )

    return state
