import functools
from typing import Callable

import jax
import jax.numpy as jnp

from rl.types import Params


def fn_parallel(fn: Callable) -> Callable:
    """args must be entered in the same order as in fn to allow vmapping"""

    def _fn(params: Params, *trees, **hyperparams):
        res = jax.tree_util.tree_map(
            functools.partial(fn, params, **hyperparams), *trees
        )

        n_outputs = len(list(res.values())[0])
        outputs = [{} for _ in range(n_outputs)]
        for a, v in res.items():
            for i in range(len(outputs)):
                outputs[i][a] = v[i]
        return outputs

    return _fn


def explore_general_factory(explore_fn: Callable, batched: bool, parallel: bool):
    def input_fn(inputs):
        if not batched:
            return jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), inputs)
        return inputs

    explore_fn = fn_parallel(explore_fn) if parallel else explore_fn

    def output_fn(outputs):
        if not batched:
            return jax.tree_map(lambda x: jnp.squeeze(x, axis=0), outputs)
        return outputs

    def fn(params: Params, key, *trees, **hyperparams):
        inputs = input_fn(trees)
        results = explore_fn(params, key, *inputs, **hyperparams)
        outputs = output_fn(results)
        return outputs

    return jax.jit(fn)


def process_experience_general_factory(
    process_experience_fn: Callable, vectorized: bool, parallel: bool
):
    def output_fn(outputs):
        if parallel:
            if vectorized:
                outputs = [jnp.concatenate(list(o.values()), axis=1) for o in outputs]
            else:
                outputs = [jnp.stack(list(o.values()), axis=1) for o in outputs]

        if vectorized:
            outputs = jax.tree_map(
                lambda x: jnp.reshape(x, (-1, *x.shape[2:])), outputs
            )

        return outputs

    def fn(*args, **kwargs):
        return output_fn(process_experience_fn(*args, **kwargs))

    return jax.jit(fn)
