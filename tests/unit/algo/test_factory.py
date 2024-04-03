import jax.numpy as jnp

from kitae.algos.factory import fn_parallel
from kitae.algos.factory import explore_general_factory


def test_fn_parallel():

    def dummy_fn(a, b, *, c):
        return a * b + c, a * b + c + 1

    a = jnp.ones((), dtype=jnp.float32)

    args = (
        {
            "0": jnp.ones((1, 2), dtype=jnp.float32),
            "1": 2 * jnp.ones((1, 2), dtype=jnp.float32),
        },
    )
    kwargs = {"c": 3 * jnp.ones((), dtype=jnp.float32)}

    res = fn_parallel(dummy_fn)(a, *args, **kwargs)

    assert res[0]["0"].shape == res[0]["1"].shape == (1, 2)
    assert res[1]["0"].shape == res[1]["1"].shape == (1, 2)
    assert jnp.array_equal(res[0]["0"], 4 * jnp.ones((1, 2)))
    assert jnp.array_equal(res[0]["1"], 5 * jnp.ones((1, 2)))
    assert jnp.array_equal(res[1]["0"], 5 * jnp.ones((1, 2)))
    assert jnp.array_equal(res[1]["1"], 6 * jnp.ones((1, 2)))


def test_explore_general_factory():

    def dummy_fn(a, b, c, d):
        return a * b * c + d, a * b * c + d + 1

    a = jnp.ones((), dtype=jnp.float32)
    d = 3 * jnp.ones((), dtype=jnp.float32)

    # test normal
    b = 2 * jnp.ones((2,), dtype=jnp.float32)
    c = 2 * jnp.ones((2,), dtype=jnp.float32)
    res = explore_general_factory(dummy_fn, False, False)(a, b, c, d=d)

    assert res[0].shape == (2,)
    assert res[1].shape == (2,)
    assert jnp.array_equal(res[0], 7 * jnp.ones((2,)))
    assert jnp.array_equal(res[1], 8 * jnp.ones((2,)))

    # test vectorized
    b = 2 * jnp.ones((1, 2), dtype=jnp.float32)
    c = 2 * jnp.ones((1, 2), dtype=jnp.float32)
    res = explore_general_factory(dummy_fn, True, False)(a, b, c, d=d)

    assert res[0].shape == (1, 2)
    assert res[1].shape == (1, 2)
    assert jnp.array_equal(res[0], 7 * jnp.ones((1, 2)))
    assert jnp.array_equal(res[1], 8 * jnp.ones((1, 2)))

    # test parallel
    b = {
        "0": 2 * jnp.ones((2,), dtype=jnp.float32),
        "1": 2 * jnp.ones((2,), dtype=jnp.float32),
    }
    c = {
        "0": 2 * jnp.ones((2,), dtype=jnp.float32),
        "1": 2 * jnp.ones((2,), dtype=jnp.float32),
    }
    res = explore_general_factory(dummy_fn, False, True)(a, b, c, d=d)

    assert res[0]["0"].shape == res[0]["1"].shape == (2,)
    assert res[1]["0"].shape == res[1]["1"].shape == (2,)
    assert jnp.array_equal(res[0]["0"], 7 * jnp.ones((2,)))
    assert jnp.array_equal(res[0]["1"], 7 * jnp.ones((2,)))
    assert jnp.array_equal(res[1]["0"], 8 * jnp.ones((2,)))
    assert jnp.array_equal(res[1]["1"], 8 * jnp.ones((2,)))

    # test vectorized parallel
    b = {
        "0": 2 * jnp.ones((1, 2), dtype=jnp.float32),
        "1": 2 * jnp.ones((1, 2), dtype=jnp.float32),
    }
    c = {
        "0": 2 * jnp.ones((1, 2), dtype=jnp.float32),
        "1": 2 * jnp.ones((1, 2), dtype=jnp.float32),
    }
    res = explore_general_factory(dummy_fn, False, True)(a, b, c, d=d)

    assert res[0]["0"].shape == res[0]["1"].shape == (1, 2)
    assert res[1]["0"].shape == res[1]["1"].shape == (1, 2)
    assert jnp.array_equal(res[0]["0"], 7 * jnp.ones((1, 2)))
    assert jnp.array_equal(res[0]["1"], 7 * jnp.ones((1, 2)))
    assert jnp.array_equal(res[1]["0"], 8 * jnp.ones((1, 2)))
    assert jnp.array_equal(res[1]["1"], 8 * jnp.ones((1, 2)))
