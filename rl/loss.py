import chex
import distrax as dx
import jax
import jax.numpy as jnp


def loss_policy_ppo(dist, log_probs, log_probs_old, gaes, clip_eps, entropy_coef):
    """PPO Policy loss function

    Args:
        logits: An Array of shape (..., N_actions)
        log_probs: An Array of shape (..., 1)
        log_probs_old: An Array of shape (..., 1)
        gaes: An Array of shape (..., 1)
        clip_eps: A float
        entropy_coef: A float

    Returns:
        policy_loss: A float
        infos: A dictionary containing information computed
            loss_policy: A float
            entropy: A Array of shape (..., N_actions)
            kl_divergence: A float
    """
    chex.assert_equal_shape([log_probs, log_probs_old, gaes])

    log_ratios = log_probs - log_probs_old
    ratios = jnp.exp(log_ratios)

    ratios_clip = jnp.clip(ratios, 1 - clip_eps, 1 + clip_eps)
    loss_policy = -jnp.mean(jnp.fmin(ratios * gaes, ratios_clip * gaes))

    entropy = dist.entropy()
    loss_entropy = -jnp.mean(entropy)

    kl_divergence = jax.lax.stop_gradient(jnp.mean((ratios - 1) - log_ratios))
    infos = {
        "loss_policy": loss_policy,
        "mean_entropy": jnp.mean(entropy),
        "kl_divergence": kl_divergence,
        "entropy": entropy,
    }

    return loss_policy + entropy_coef * loss_entropy, infos


def loss_value_clip(
    values: jax.Array, targets: jax.Array, values_old: jax.Array, clip_eps: float
) -> tuple[float, dict[str, jax.Array]]:
    """Clipped value loss function

    A clipped value loss ensures smaller updates of the ValueModule.

    Args:
        values: An Array of shape (..., 1)
        targets: An Array of shape (..., 1)
        values_old: An Array of shape (..., 1)
        clip_eps: A float

    Returns:
        value_loss: A float
        infos: A dictionary containing information computed
            loss_value: A float
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


def loss_mean_squared_error(x: jax.Array, y: jax.Array) -> float:
    chex.assert_equal_shape([x, y])
    return jnp.mean(jnp.square(x - y))
