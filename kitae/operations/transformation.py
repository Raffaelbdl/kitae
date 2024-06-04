from gymnasium.spaces import Box
import jax
import jax.numpy as jnp


def linear_interpolation(x: jax.Array, a: float, b: float):
    x = jnp.fmax(jnp.fmin(x, 1.0), 0.0)
    return x * (b - a) + a


def inverse_linear_interpolation(y: jax.Array, a: float, b: float):
    y = jnp.fmax(jnp.fmin(y, b), a)
    return (y - a) / (b - a)


def normalize_frames(x: jax.Array) -> jax.Array:
    return linear_interpolation(x / 255.0, -1.0, 1.0)


def flatten(x: jax.Array) -> jax.Array:
    return jnp.reshape(x, (-1,))


def action_clip(x: jax.Array, action_space: Box) -> jax.Array:
    return jnp.clip(x, action_space.low, action_space.high)
