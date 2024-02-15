from typing import Callable

from flax import struct
import jax


class UpdateModule(struct.PyTreeNode):
    update_fn: Callable = struct.field(pytree_node=False)
    state: struct.PyTreeNode = struct.field(pytree_node=True)


def update_pipeline(
    update_modules: list[UpdateModule],
    key: jax.Array,
    batch: tuple,
) -> list[UpdateModule]:
    info = {}

    for i, module in enumerate(update_modules):
        key, _key = jax.random.split(key, 2)
        state, module_info = module.update_fn(module.state, _key, batch)

        update_modules[i] = module.replace(state=state)
        info |= module_info

    return update_modules, info
