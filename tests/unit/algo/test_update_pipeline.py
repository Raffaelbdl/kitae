from collections import namedtuple

import jax
import jax.numpy as jnp

from rl_tools.algos.pipeline import UpdateModule
from rl_tools.algos.pipeline import update_pipeline

ExperienceNamedTuple = namedtuple("ExperienceNamedTuple", ["field_0", "field_1"])


def test_update_pipeline():
    UpdateModuleA = UpdateModule(
        update_fn=lambda s, k, b: (s + b[0], {"k": 0}), state=0
    )
    UpdateModuleB = UpdateModule(
        update_fn=lambda s, k, b: (s + b[1], {"k": 1}), state=0
    )

    update_modules = [UpdateModuleA, UpdateModuleB]
    key = jax.random.key(0)
    batch = (0, 1)

    update_modules, info = update_pipeline(update_modules, key, batch)
    assert update_modules[0].state == 0
    assert update_modules[1].state == 1
    assert info["k"] == 1
