import functools
from typing import Any, Callable

import jax
import jax.numpy as jnp


def fn_parallel(fn: Callable) -> Callable:
    """Parallelizes a function for mutliple agents.

    Typical usage for function with args:
        - state
        - trees of structure: {"agent_0": Array, "agent_1": Array, ...}
        - hyperparameters

    The wrapped function returns a list of trees with the same structure as input.

    Warning: args must be entered in the same order as in fn to allow vmapping.
    """

    def wrapped(state: Any, *trees, **hyperparams):
        results = jax.tree_util.tree_map(
            functools.partial(fn, state, **hyperparams), *trees
        )

        # transform the structure of results
        # output = {"agent_0": [out1, out2], "agent_1": [out1, out2]}
        # -> [{"agent_1": out1, "agent_2": out1}, {"agent_1": out2, "agent_2": out2}]

        n_outputs = len(list(results.values())[0])
        outputs = [{} for _ in range(n_outputs)]

        for key, value in results.items():
            for out_pos in range(n_outputs):
                outputs[out_pos][key] = value[out_pos]

        return outputs

    return wrapped


def explore_general_factory(
    explore_fn: Callable, vectorized: bool, parallel: bool
) -> Callable:
    """Generalizes a explore_fn to vector and parallel envs."""

    def input_fn(inputs):
        if not vectorized:
            return jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), inputs)
        return inputs

    explore_fn = fn_parallel(explore_fn) if parallel else explore_fn

    def output_fn(outputs):
        if not vectorized:
            return jax.tree_map(lambda x: jnp.squeeze(x, axis=0), outputs)
        return outputs

    def general_fn(state: Any, key: jax.Array, *trees, **hyperparams):
        inputs = input_fn(trees)
        results = explore_fn(state, key, *inputs, **hyperparams)
        outputs = output_fn(results)
        return outputs

    return general_fn
