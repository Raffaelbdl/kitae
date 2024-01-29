import chex

from rl.modules.modules import TrainState


@chex.dataclass
class TrainStatePolicyQValue:
    """Jittable collection of policy and qvalue TrainState.

    Contrary to on-policy algorithms, off-policy algorithms often require
    both separate encoders and separate updates of policy and qvalue modules.
    This class store them separately, in a chex.dataclass so that it can be
    used in jitted function.

    Attributes:
        policy_state, qvalue_state:
            Custom TrainState that hold a `target_params` field.
    """

    policy_state: TrainState
    qvalue_state: TrainState


@chex.dataclass
class TrainStatePolicyQValueTemperature:
    """Jittable collection of policy and qvalue TrainState.

    Contrary to on-policy algorithms, off-policy algorithms often require
    both separate encoders and separate updates of policy and qvalue modules.
    This class store them separately, in a chex.dataclass so that it can be
    used in jitted function.

    Attributes:
        policy_state, qvalue_state:
            Custom TrainState that hold a `target_params` field.
        temperature_state: TrainState
    """

    policy_state: TrainState
    qvalue_state: TrainState
    temperature_state: TrainState
