from flax.struct import field, dataclass, dataclass_transform


@dataclass_transform(field_specifiers=(field,))
class AgentPyTree:
    def __init_subclass__(cls):
        dataclass(cls, frozen=False)
