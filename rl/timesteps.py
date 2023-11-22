"""Timesteps operations useful for reinforcement learning"""
import chex
import jax
import jax.numpy as jnp


def calculate_gaes_targets(
    values: jax.Array,
    next_values: jax.Array,
    discounts: jax.Array,
    rewards: jax.Array,
    _lambda: float,
    normalize: bool,
) -> tuple[jax.Array, jax.Array]:
    """Calculates general advantage estimations

    Args:
        values: An Array of shape (T, 1)
        next_values: An Array of shape (T, 1)
        discounts: An Array of shape (T, 1)
        rewards: An Array of shape (T, 1)
        _lambda: A float
        normalize: A boolean indicating if the advantages should be normalized
    Returns:
        gaes: An array of shape (T, 1)

    values are estimated by using observations.
    next_values are estimated by using next_observations.
    discounts are calculated using dones and the discount factor.
    """
    chex.assert_equal_shape([values, next_values, discounts, rewards])
    chex.assert_rank([values, next_values, discounts, rewards], 2)

    td_errors = rewards + discounts * next_values - values

    def _fn(acc, xs):
        tds, discs = xs
        acc = tds + discs * _lambda * acc
        return acc, acc

    init = jnp.zeros((1,))
    xs = [td_errors, discounts]
    _, gaes = jax.lax.scan(_fn, init, xs, reverse=True)

    targets = gaes + values
    if normalize:
        gaes = gaes - jnp.mean(gaes) / (jnp.std(gaes) + 1e-8)

    return gaes, targets
