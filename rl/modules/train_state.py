from typing import Callable

import chex
from flax import struct
from flax.training import train_state

from rl.types import Params


class TrainState(train_state.TrainState):
    target_params: Params = None


@chex.dataclass
class PolicyValueTrainState:
    policy_state: TrainState
    value_state: TrainState
    encoder_state: TrainState


@chex.dataclass
class PolicyQValueTrainState:
    policy_state: TrainState
    qvalue_state: TrainState
