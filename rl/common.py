import jax


def ensure_int(action: jax.Array | int) -> int:
    return int(action)
