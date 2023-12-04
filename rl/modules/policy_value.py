import functools
from typing import Callable, Type

import chex
from einops import rearrange
import flax
from flax import struct
from flax import linen as nn
from flax.training.train_state import TrainState
from gymnasium import spaces
import jax
from jax import numpy as jnp
import ml_collections
import numpy as np
import optax

from rl.base import Params
from rl.modules.modules import modules_factory, create_params


@chex.dataclass
class ParamsPolicyValue:
    params_encoder: Params
    params_policy: Params
    params_value: Params


class TrainStatePolicyValue(TrainState):
    encoder_fn: Callable = struct.field(pytree_node=False)
    policy_fn: Callable = struct.field(pytree_node=False)
    value_fn: Callable = struct.field(pytree_node=False)


def create_modules(
    observation_space: spaces.Space,
    action_space: spaces.Discrete,
    shared_encoder: bool,
    *,
    rearrange_pattern: str,
) -> dict[str, nn.Module]:
    return modules_factory(
        observation_space,
        action_space,
        shared_encoder,
        rearrange_pattern=rearrange_pattern,
    )


def create_params_policy_value(
    key: jax.Array,
    policy: nn.Module,
    value: nn.Module,
    encoder: nn.Module,
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
            params_encoder=create_params(
                key1, encoder, observation_space.shape, tabulate=tabulate
            ),
            params_policy=create_params(key2, policy, hidden_shape, tabulate=tabulate),
            params_value=create_params(key3, value, hidden_shape, tabulate=tabulate),
        )
    else:
        return ParamsPolicyValue(
            params_encoder=create_params(
                key1, encoder, observation_space.shape, tabulate=tabulate
            ),
            params_policy=create_params(
                key2, policy, observation_space.shape, tabulate=tabulate
            ),
            params_value=create_params(
                key3, value, observation_space.shape, tabulate=tabulate
            ),
        )


def create_train_state_policy_value(
    policy: nn.Module,
    value: nn.Module,
    encoder: nn.Module,
    params: ParamsPolicyValue,
    config: ml_collections.ConfigDict,
    *,
    n_envs: int = 1,
) -> TrainStatePolicyValue:
    num_batches = n_envs * config.max_buffer_size // config.batch_size
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
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate, eps=1e-5),
    )

    return TrainStatePolicyValue.create(
        apply_fn=None,
        params=params,
        tx=tx,
        policy_fn=policy.apply,
        value_fn=value.apply,
        encoder_fn=encoder.apply,
    )


def train_state_policy_value_factory(
    key: jax.Array,
    config: ml_collections,
    *,
    rearrange_pattern: str,
    n_envs: int,
    tabulate: bool = False,
) -> TrainStatePolicyValue:
    modules = create_modules(
        config.observation_space,
        config.action_space,
        config.shared_encoder,
        rearrange_pattern=rearrange_pattern,
    )
    params = create_params_policy_value(
        key,
        modules["policy"],
        modules["value"],
        modules["encoder"],
        config.observation_space,
        shared_encoder=config.shared_encoder,
        tabulate=tabulate,
    )
    state = create_train_state_policy_value(
        modules["policy"],
        modules["value"],
        modules["encoder"],
        params,
        config,
        n_envs=n_envs,
    )
    return state
