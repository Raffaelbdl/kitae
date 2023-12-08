from typing import Callable

from flax import linen as nn
from flax.training.train_state import TrainState
from gymnasium import spaces
import jax
import ml_collections
import optax

from rl.modules.modules import modules_factory, create_params
from rl.modules.optimizer import linear_learning_rate_schedule

from rl.types import Params


def create_modules(
    observation_space: spaces.Space,
    action_space: spaces.Discrete,
    *,
    rearrange_pattern: str,
    preprocess_fn: Callable,
) -> dict[str, nn.Module]:
    modules = modules_factory(
        observation_space,
        action_space,
        False,
        rearrange_pattern=rearrange_pattern,
        preprocess_fn=preprocess_fn,
    )
    return {"qvalue": modules["policy"]}


def create_params_qvalue(
    key: jax.Array,
    qvalue: nn.Module,
    observation_space: spaces.Space,
    *,
    tabulate: bool = False,
) -> Params:
    return create_params(key, qvalue, observation_space.shape, tabulate=tabulate)


def create_train_state_qvalue(
    qvalue: nn.Module,
    params: Params,
    config: ml_collections.ConfigDict,
    *,
    n_envs: int = 1,
) -> TrainState:
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

    return TrainState.create(apply_fn=qvalue.apply, params=params, tx=tx)


def train_state_qvalue_factory(
    key: jax.Array,
    config: ml_collections.ConfigDict,
    *,
    rearrange_pattern: str,
    preprocess_fn: Callable,
    tabulate: bool = False,
) -> TrainState:
    modules = create_modules(
        config.env_config.observation_space,
        config.env_config.action_space,
        rearrange_pattern=rearrange_pattern,
        preprocess_fn=preprocess_fn,
    )
    params = create_params_qvalue(
        key, modules["qvalue"], config.env_config.observation_space, tabulate=tabulate
    )
    state = create_train_state_qvalue(
        modules["qvalue"], params, config, n_envs=config.env_config.n_envs
    )
    return state
