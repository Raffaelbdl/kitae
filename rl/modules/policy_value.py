from typing import Callable

import chex
from flax import struct
from flax import linen as nn
from flax.training.train_state import TrainState
from gymnasium import spaces
import jax
import ml_collections
import optax

from rl.modules.modules import modules_factory, create_params
from rl.modules.optimizer import linear_learning_rate_schedule

from rl.types import Params


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
    preprocess_fn: Callable,
) -> dict[str, nn.Module]:
    return modules_factory(
        observation_space,
        action_space,
        shared_encoder,
        rearrange_pattern=rearrange_pattern,
        preprocess_fn=preprocess_fn,
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
    learning_rate = config.learning_rate

    if config.learning_rate_annealing:
        learning_rate = linear_learning_rate_schedule(
            learning_rate,
            0.0,
            n_envs=n_envs,
            n_env_steps=config.n_env_steps,
            max_buffer_size=config.max_buffer_size,
            batch_size=config.batch_size,
            num_epochs=config.num_epochs,
        )

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
    preprocess_fn: Callable,
    n_envs: int,
    tabulate: bool = False,
) -> TrainStatePolicyValue:
    modules = create_modules(
        config.observation_space,
        config.action_space,
        config.shared_encoder,
        rearrange_pattern=rearrange_pattern,
        preprocess_fn=preprocess_fn,
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


def train_state_policy_value_population_factory(
    key: jax.Array,
    config: ml_collections,
    population_size: int,
    *,
    rearrange_pattern: str,
    preprocess_fn: Callable,
    n_envs: int,
    tabulate: bool = False,
) -> TrainStatePolicyValue:
    modules = create_modules(
        config.observation_space,
        config.action_space,
        config.shared_encoder,
        rearrange_pattern=rearrange_pattern,
        preprocess_fn=preprocess_fn,
    )
    params_population = [
        create_params_policy_value(
            key,
            modules["policy"],
            modules["value"],
            modules["encoder"],
            config.observation_space,
            shared_encoder=config.shared_encoder,
            tabulate=(i == 0 and tabulate),
        )
        for i in range(population_size)
    ]
    state = create_train_state_policy_value(
        modules["policy"],
        modules["value"],
        modules["encoder"],
        params_population,
        config,
        n_envs=n_envs,
    )
    return state
