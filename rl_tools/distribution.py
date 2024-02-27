import chex
import distrax as dx
import jax
from jax import numpy as jnp


def get_log_probs(dist: dx.Distribution, actions: jax.Array, log_probs_old: jax.Array):
    if isinstance(dist, dx.Normal):
        chex.assert_rank(actions, 2)
        chex.assert_rank(log_probs_old, 2)
        return (
            jnp.sum(dist.log_prob(actions), axis=-1, keepdims=True),
            jnp.sum(log_probs_old, axis=-1, keepdims=True),
        )
    elif isinstance(dist, dx.Categorical):
        chex.assert_rank(actions, 1)
        return (
            jnp.expand_dims(dist.log_prob(actions), axis=-1),
            jnp.expand_dims(log_probs_old, axis=-1),
        )
    else:
        raise NotImplementedError
