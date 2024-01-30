from typing import Callable

import chex
from flax import struct
from flax.training import train_state

from rl.types import Params


class TrainState(train_state.TrainState):
    target_params: Params = None


class PolicyValueTrainState(train_state.TrainState):
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


@chex.dataclass
class PolicyQValueTrainState:
    policy_state: TrainState
    qvalue_state: TrainState
