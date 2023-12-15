from dataclasses import dataclass
from typing import Callable

from flax import linen as nn
from flax.training.train_state import TrainState
from gymnasium import spaces
import jax
import ml_collections
import optax

from rl.config import AlgoConfig
from rl.modules.modules import init_params, encoder_factory, PassThrough
from rl.modules.optimizer import linear_learning_rate_schedule

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


# I keep a dataclass so that I can reuse the
# code for continuous action space Q learning
@dataclass
class QModules:
    encoder: nn.Module
    qvalue: nn.Module


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
