import jax
import jax.numpy as jnp


def update(state: jax.Array, batch: jax.Array) -> tuple[jax.Array, dict]:
    state += batch
    a = jnp.mean(batch)
    return state, {"a": a}


def main():
    batches = jnp.array(
        [
            1 * jnp.ones((32,)),
            2 * jnp.ones((32,)),
            3 * jnp.ones((32,)),
        ]
    )
    state, info = jnp.zeros((32,)), {}
    state, info = jax.lax.scan(update, state, batches)
    print(state)
    print(info)


if __name__ == "__main__":
    main()
