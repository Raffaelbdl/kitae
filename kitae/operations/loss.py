"""Collection of loss functions for reinforcement learning."""

import chex
import distrax as dx
import jax
import jax.numpy as jnp

from kitae.types import LossDict


def loss_policy_ppo(
    dist: dx.Distribution,
    log_probs: jax.Array,
    log_probs_old: jax.Array,
    gaes: jax.Array,
    clip_eps: float,
    entropy_coef: float,
) -> tuple[float, LossDict]:
    """Proximal Policy Optimization's policy loss function.

    Args:
        dist: A dx.Distribution to compute the entropy
        logits: An Array of shape (..., N_actions)
        log_probs: An Array of shape (..., 1)
        log_probs_old: An Array of shape (..., 1)
        gaes: An Array of shape (..., 1)
        clip_eps: A float
        entropy_coef: A float

    Returns:
        A float corresponding to the loss value.
        A LossDict with the following keys: `["loss_policy", "entropy", "kl_divergence"]`
    """
    chex.assert_equal_shape([log_probs, log_probs_old, gaes])

    log_ratios = log_probs - log_probs_old
    ratios = jnp.exp(log_ratios)

    ratios_clip = jnp.clip(ratios, 1 - clip_eps, 1 + clip_eps)
    loss_policy = -jnp.mean(jnp.fmin(ratios * gaes, ratios_clip * gaes))

    entropy = dist.entropy()
    loss_entropy = -jnp.mean(entropy)

    kl_divergence = jax.lax.stop_gradient(jnp.mean((ratios - 1) - log_ratios))
    info = {
        "loss_policy": loss_policy,
        "mean_entropy": jnp.mean(entropy),
        "kl_divergence": kl_divergence,
    }

    return loss_policy + entropy_coef * loss_entropy, info


def loss_value_clip(
    values: jax.Array,
    targets: jax.Array,
    values_old: jax.Array,
    clip_eps: float,
) -> tuple[float, LossDict]:
    """Clipped value loss function

    A clipped value loss ensures smaller updates of the value.

    Args:
        values: An Array of shape (..., 1)
        targets: An Array of shape (..., 1)
        values_old: An Array of shape (..., 1)
        clip_eps: A float

    Returns:
        A float corresponding to the loss value.
        A LossDict with the following keys: `["loss_value"]`
    """
    chex.assert_equal_shape([values, values_old, targets])

    values_clip = jnp.clip(values - values_old, -clip_eps, clip_eps)
    values_clip += values_old

    loss_value_unclip = jnp.square(values - targets)
    loss_value_clip = jnp.square(values_clip - targets)
    loss_value = jnp.mean(jnp.fmax(loss_value_unclip, loss_value_clip))

    infos = {"loss_value": loss_value}

    return loss_value, infos


def loss_shannon_jensen_divergence(
    average_logits: jax.Array, average_entropy: jax.Array
) -> float:
    """Shannon Jensen Divergence loss function

    Shannon Jensen Divergence loss is used to increase
    the behaviour diversity of a population.

    * Compute average_logits by averaging logits over the last axis
    * Compute average_entropy by averaging entropies over the last axis

    Args:
        average_logits: An Array of shape (..., N_actions)
        average_entropy: An Array of shape (..., N_actions)
    Returns:
        shannon_jensen_divergence_loss: A float
    """
    chex.assert_equal_rank([average_logits, average_entropy])
    return -jnp.mean(dx.Categorical(logits=average_logits).entropy() - average_entropy)
