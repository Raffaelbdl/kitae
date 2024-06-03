from collections import namedtuple
import pytest

import jax
import jax.numpy as jnp

from kitae.algos.experience import merge_n_first_dims
from kitae.algos.experience import stack_and_merge_n_first_dims
from kitae.algos.experience import tuple_to_dict
from kitae.algos.experience import dict_to_tuple

from kitae.algos.experience import ExperiencePipeline


def test_merge_n_first_dims():
    foo_0 = jnp.zeros((10,))
    foo_1 = jnp.zeros((10, 5, 2))

    def test_non_jit():
        with pytest.raises(AssertionError):
            merge_n_first_dims(foo_0, -1)

        with pytest.raises(AssertionError):
            merge_n_first_dims(foo_0, 0)

        _foo = merge_n_first_dims(foo_0, 1)
        assert _foo.shape == (10,)

        _foo = merge_n_first_dims(foo_0, 2)
        assert _foo.shape == (10,)

        foo = jnp.zeros((10, 5, 2))
        _foo = merge_n_first_dims(foo_1, 2)
        assert _foo.shape == (50, 2)
        _foo = merge_n_first_dims(foo_1, 3)
        assert _foo.shape == (100,)

    test_non_jit()

    def test_jit():
        _merge_n_first_dims = jax.jit(merge_n_first_dims, static_argnums=(1,))

        with pytest.raises(AssertionError):
            _merge_n_first_dims(foo_0, -1)

        with pytest.raises(AssertionError):
            _merge_n_first_dims(foo_0, 0)

        _foo = _merge_n_first_dims(foo_0, 1)
        assert _foo.shape == (10,)

        _foo = _merge_n_first_dims(foo_0, 2)
        assert _foo.shape == (10,)

        _foo = _merge_n_first_dims(foo_1, 2)
        assert _foo.shape == (50, 2)
        _foo = _merge_n_first_dims(foo_1, 3)
        assert _foo.shape == (100,)

    test_jit()


def test_stack_and_merge_n_first_dims():
    foo_0 = [jnp.zeros((10,)), jnp.zeros((10,))]
    foo_1 = [jnp.zeros((10, 5, 2)), jnp.zeros((10, 5, 2))]

    def test_non_jit():
        with pytest.raises(AssertionError):
            stack_and_merge_n_first_dims(foo_0, -1)

        with pytest.raises(AssertionError):
            stack_and_merge_n_first_dims(foo_0, 0)

        _foo = stack_and_merge_n_first_dims(foo_0, 1)
        assert _foo.shape == (10, 2)

        _foo = stack_and_merge_n_first_dims(foo_0, 2)
        assert _foo.shape == (20,)

        _foo = stack_and_merge_n_first_dims(foo_1, 2)
        assert _foo.shape == (20, 5, 2)
        _foo = stack_and_merge_n_first_dims(foo_1, 3)
        assert _foo.shape == (100, 2)

    test_non_jit()

    def test_jit():
        _stack_and_merge_n_first_dims = jax.jit(
            stack_and_merge_n_first_dims, static_argnums=(1,)
        )

        with pytest.raises(AssertionError):
            _stack_and_merge_n_first_dims(foo_0, -1)

        with pytest.raises(AssertionError):
            _stack_and_merge_n_first_dims(foo_0, 0)

        _foo = _stack_and_merge_n_first_dims(foo_0, 1)
        assert _foo.shape == (10, 2)

        _foo = _stack_and_merge_n_first_dims(foo_0, 2)
        assert _foo.shape == (20,)

        _foo = _stack_and_merge_n_first_dims(foo_1, 2)
        assert _foo.shape == (20, 5, 2)
        _foo = _stack_and_merge_n_first_dims(foo_1, 3)
        assert _foo.shape == (100, 2)

    test_jit()


def test_tuple_to_dict():
    FooTuple = namedtuple("FooTuple", ["a", "b"])
    foo = FooTuple(
        a={"a": jnp.zeros((10,)), "b": jnp.zeros((10,)) + 1},
        b={"a": jnp.zeros((10, 5, 2)) + 2, "b": jnp.zeros((10, 5, 2)) + 3},
    )

    _foo = tuple_to_dict(foo)
    assert isinstance(_foo, dict)
    assert jnp.array_equal(_foo["a"].a, jnp.zeros((10,)))
    assert jnp.array_equal(_foo["a"].b, jnp.zeros((10, 5, 2)) + 2)
    assert jnp.array_equal(_foo["b"].a, jnp.zeros((10,)) + 1)
    assert jnp.array_equal(_foo["b"].b, jnp.zeros((10, 5, 2)) + 3)


def test_dict_to_tuple():
    FooTuple = namedtuple("FooTuple", ["a", "b"])
    foo = {
        "a": FooTuple(a=jnp.zeros((10,)), b=jnp.zeros((10, 5, 2)) + 2),
        "b": FooTuple(a=jnp.zeros((10,)) + 1, b=jnp.zeros((10, 5, 2)) + 3),
    }

    _foo = dict_to_tuple(foo)
    assert isinstance(_foo, FooTuple)
    assert jnp.array_equal(_foo.a["a"], jnp.zeros((10,)))
    assert jnp.array_equal(_foo.b["a"], jnp.zeros((10, 5, 2)) + 2)
    assert jnp.array_equal(_foo.a["b"], jnp.zeros((10,)) + 1)
    assert jnp.array_equal(_foo.b["b"], jnp.zeros((10, 5, 2)) + 3)


# region Test ExperiencePipeline
ExperienceNamedTuple = namedtuple("ExperienceNamedTuple", ["field_0", "field_1"])


def transform_field_0(
    state, key, experience: ExperienceNamedTuple
) -> ExperienceNamedTuple:
    jax.random.split(key)  # ensures that key has right shape
    experience = experience._replace(field_0=experience.field_0 + 1)
    return experience


def transform_field_1(
    state, key, experience: ExperienceNamedTuple
) -> ExperienceNamedTuple:
    experience = experience._replace(field_1=experience.field_1 + 2)
    return experience


def test_experience_pipeline():
    key = jax.random.key(0)

    foo = ExperienceNamedTuple(
        field_0=jnp.zeros((10, 5)), field_1=jnp.zeros((10, 5, 2))
    )
    ma_foo = ExperienceNamedTuple(
        field_0={"a": jnp.zeros((10, 5)), "b": jnp.zeros((10, 5)) + 1},
        field_1={"a": jnp.zeros((10, 5, 2)) + 2, "b": jnp.zeros((10, 5, 2)) + 3},
    )

    def test_no_parallel_no_vectorized():
        pipeline = ExperiencePipeline(
            transforms=[transform_field_0, transform_field_1],
            vectorized=False,
            parallel=False,
        )

        _foo: ExperienceNamedTuple = pipeline.run(None, key, foo)
        assert jnp.array_equal(_foo.field_0, jnp.zeros((10, 5)) + 1)
        assert jnp.array_equal(_foo.field_1, jnp.zeros((10, 5, 2)) + 2)

        __foo: ExperienceNamedTuple = pipeline.run_single_pipe(None, key, foo)
        assert jnp.array_equal(_foo.field_0, __foo.field_0)
        assert jnp.array_equal(_foo.field_1, __foo.field_1)

        _foo: ExperienceNamedTuple = jax.jit(pipeline.run)(None, key, foo)
        assert jnp.array_equal(_foo.field_0, jnp.zeros((10, 5)) + 1)
        assert jnp.array_equal(_foo.field_1, jnp.zeros((10, 5, 2)) + 2)

        __foo: ExperienceNamedTuple = jax.jit(pipeline.run_single_pipe)(None, key, foo)
        assert jnp.array_equal(_foo.field_0, __foo.field_0)
        assert jnp.array_equal(_foo.field_1, __foo.field_1)

        with pytest.raises(TypeError):
            pipeline.run(None, key, ma_foo)
        with pytest.raises(TypeError):
            pipeline.run_single_pipe(None, key, ma_foo)
        with pytest.raises(TypeError):
            jax.jit(pipeline.run)(None, key, ma_foo)
        with pytest.raises(TypeError):
            jax.jit(pipeline.run_single_pipe)(None, key, ma_foo)

    test_no_parallel_no_vectorized()

    def test_parallel_no_vectorized():
        pipeline = ExperiencePipeline(
            transforms=[transform_field_0, transform_field_1],
            vectorized=False,
            parallel=True,
        )

        _field_0 = jnp.reshape(
            jnp.stack([jnp.zeros((10, 5)) + 1, jnp.zeros((10, 5)) + 2], axis=1), (20, 5)
        )
        _field_1 = jnp.reshape(
            jnp.stack([jnp.zeros((10, 5, 2)) + 4, jnp.zeros((10, 5, 2)) + 5], axis=1),
            (20, 5, 2),
        )

        _foo: ExperienceNamedTuple = pipeline.run(None, key, ma_foo)
        assert jnp.array_equal(_foo.field_0, _field_0)
        assert jnp.array_equal(_foo.field_1, _field_1)

        __foo: ExperienceNamedTuple = pipeline.run_parallel(None, key, ma_foo)
        assert jnp.array_equal(_foo.field_0, _field_0)
        assert jnp.array_equal(_foo.field_1, _field_1)

        _foo: ExperienceNamedTuple = jax.jit(pipeline.run)(None, key, ma_foo)
        assert jnp.array_equal(_foo.field_0, _field_0)
        assert jnp.array_equal(_foo.field_1, _field_1)

        __foo: ExperienceNamedTuple = jax.jit(pipeline.run_parallel)(None, key, ma_foo)
        assert jnp.array_equal(_foo.field_0, __foo.field_0)
        assert jnp.array_equal(_foo.field_1, __foo.field_1)

        with pytest.raises(AttributeError):
            pipeline.run(None, key, foo)
        with pytest.raises(AttributeError):
            pipeline.run_parallel(None, key, foo)
        with pytest.raises(AttributeError):
            jax.jit(pipeline.run)(None, key, foo)
        with pytest.raises(AttributeError):
            jax.jit(pipeline.run_parallel)(None, key, foo)

    test_parallel_no_vectorized()

    def test_no_parallel_vectorized():
        pipeline = ExperiencePipeline(
            transforms=[transform_field_0, transform_field_1],
            vectorized=True,
            parallel=False,
        )

        _foo: ExperienceNamedTuple = pipeline.run(None, key, foo)
        assert jnp.array_equal(_foo.field_0, jnp.zeros((50,)) + 1)
        assert jnp.array_equal(_foo.field_1, jnp.zeros((50, 2)) + 2)

        __foo: ExperienceNamedTuple = pipeline.run_vectorized(None, key, foo)
        assert jnp.array_equal(_foo.field_0, __foo.field_0)
        assert jnp.array_equal(_foo.field_1, __foo.field_1)

        _foo: ExperienceNamedTuple = jax.jit(pipeline.run)(None, key, foo)
        assert jnp.array_equal(_foo.field_0, jnp.zeros((50,)) + 1)
        assert jnp.array_equal(_foo.field_1, jnp.zeros((50, 2)) + 2)

        __foo: ExperienceNamedTuple = jax.jit(pipeline.run_vectorized)(None, key, foo)
        assert jnp.array_equal(_foo.field_0, __foo.field_0)
        assert jnp.array_equal(_foo.field_1, __foo.field_1)

        with pytest.raises(AttributeError):
            pipeline.run(None, key, ma_foo)
        with pytest.raises(AttributeError):
            pipeline.run_vectorized(None, key, ma_foo)
        with pytest.raises(AttributeError):
            jax.jit(pipeline.run)(None, key, ma_foo)
        with pytest.raises(AttributeError):
            jax.jit(pipeline.run_vectorized)(None, key, ma_foo)

    test_no_parallel_vectorized()

    def test_parallel_vectorized():
        pipeline = ExperiencePipeline(
            transforms=[transform_field_0, transform_field_1],
            vectorized=True,
            parallel=True,
        )

        _field_0 = jnp.reshape(
            jnp.stack([jnp.zeros((10, 5)) + 1, jnp.zeros((10, 5)) + 2], axis=1), (100,)
        )
        _field_1 = jnp.reshape(
            jnp.stack([jnp.zeros((10, 5, 2)) + 4, jnp.zeros((10, 5, 2)) + 5], axis=1),
            (100, 2),
        )

        _foo: ExperienceNamedTuple = pipeline.run(None, key, ma_foo)
        assert jnp.array_equal(_foo.field_0, _field_0)
        assert jnp.array_equal(_foo.field_1, _field_1)

        __foo: ExperienceNamedTuple = pipeline.run_parallel_vectorized(
            None, key, ma_foo
        )
        assert jnp.array_equal(_foo.field_0, __foo.field_0)
        assert jnp.array_equal(_foo.field_1, __foo.field_1)

        _foo: ExperienceNamedTuple = jax.jit(pipeline.run)(None, key, ma_foo)
        assert jnp.array_equal(_foo.field_0, _field_0)
        assert jnp.array_equal(_foo.field_1, _field_1)

        __foo: ExperienceNamedTuple = jax.jit(pipeline.run_parallel_vectorized)(
            None, key, ma_foo
        )
        assert jnp.array_equal(_foo.field_0, __foo.field_0)
        assert jnp.array_equal(_foo.field_1, __foo.field_1)

        with pytest.raises(AttributeError):
            pipeline.run(None, key, foo)
        with pytest.raises(AttributeError):
            pipeline.run_parallel_vectorized(None, key, foo)
        with pytest.raises(AttributeError):
            jax.jit(pipeline.run)(None, key, foo)
        with pytest.raises(AttributeError):
            jax.jit(pipeline.run_parallel_vectorized)(None, key, foo)

    test_parallel_vectorized()


# endregion

if __name__ == "__main__":
    test_experience_pipeline()
