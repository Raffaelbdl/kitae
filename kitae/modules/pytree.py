from flax.core import FrozenDict
from flax.struct import field, dataclass, dataclass_transform
from flax.training import train_state


@dataclass_transform(field_specifiers=(field,))
class AgentPyTree:
    """Default Agent State class.

    Contrary to Flax's PyTreeNode, an AgentPyTree is mutable.
    """

    def __init_subclass__(cls):
        dataclass(cls, frozen=False)


class TrainState(train_state.TrainState):
    """Modified TrainState with `target_params` attribute."""

    target_params: FrozenDict = None
