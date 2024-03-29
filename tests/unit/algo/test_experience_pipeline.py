from collections import namedtuple

import jax
import jax.numpy as jnp

from rl_tools.algos.experience import ExperienceTransform
from rl_tools.algos.experience import process_experience_pipeline_factory

ExperienceNamedTuple = namedtuple("ExperienceNamedTuple", ["field_0", "field_1"])


def increment_field_0(experience: ExperienceNamedTuple) -> ExperienceNamedTuple:
    experience = experience._replace(field_0=experience.field_0 + 1)
    return experience


def increment_field_1(experience: ExperienceNamedTuple) -> ExperienceNamedTuple:
    experience = experience._replace(field_1=experience.field_1 + 1)
    return experience


ExperienceTransformA = ExperienceTransform(
    process_experience_fn=lambda s, k, e: increment_field_0(e),
    state=None,
)
ExperienceTransformB = ExperienceTransform(
    process_experience_fn=lambda s, k, e: increment_field_1(e),
    state=None,
)


def test_process_experience_pipeline_factory():
    key = jax.random.key(0)
    process_experience_pipeline = process_experience_pipeline_factory(
        False, False, ExperienceNamedTuple
    )

    experience_transforms = [ExperienceTransformA, ExperienceTransformB]
    experience = process_experience_pipeline(
        experience_transforms,
        key,
        ExperienceNamedTuple(0, 5),
    )
    assert experience.field_0 == 1
    assert experience.field_1 == 6

    experience = jax.jit(process_experience_pipeline)(
        experience_transforms,
        key,
        ExperienceNamedTuple(0, 5),
    )
    assert experience.field_0 == 1
    assert experience.field_1 == 6

    experience = process_experience_pipeline(
        experience_transforms,
        key,
        ExperienceNamedTuple(jnp.array([0, 0]), jnp.array([5, 5])),
    )
    assert jnp.all(jnp.equal(experience.field_0, jnp.array([1, 1])))
    assert jnp.all(jnp.equal(experience.field_1, jnp.array([6, 6])))

    experience = jax.jit(process_experience_pipeline)(
        experience_transforms,
        key,
        ExperienceNamedTuple(jnp.array([0, 0]), jnp.array([5, 5])),
    )
    assert jnp.all(jnp.equal(experience.field_0, jnp.array([1, 1])))
    assert jnp.all(jnp.equal(experience.field_1, jnp.array([6, 6])))

    experience = process_experience_pipeline(
        experience_transforms,
        key,
        [
            ExperienceNamedTuple(jnp.array([0, 0]), jnp.array([5, 5])),
            ExperienceNamedTuple(jnp.array([0, 0]), jnp.array([5, 5])),
        ],
    )
    assert jnp.all(jnp.equal(experience.field_0, jnp.array([[1, 1], [1, 1]])))
    assert jnp.all(jnp.equal(experience.field_1, jnp.array([[6, 6], [6, 6]])))

    experience = jax.jit(process_experience_pipeline)(
        experience_transforms,
        key,
        [
            ExperienceNamedTuple(jnp.array([0, 0]), jnp.array([5, 5])),
            ExperienceNamedTuple(jnp.array([0, 0]), jnp.array([5, 5])),
        ],
    )
    assert jnp.all(jnp.equal(experience.field_0, jnp.array([[1, 1], [1, 1]])))
    assert jnp.all(jnp.equal(experience.field_1, jnp.array([[6, 6], [6, 6]])))

    process_experience_pipeline = process_experience_pipeline_factory(
        True, False, ExperienceNamedTuple
    )
    experience = process_experience_pipeline(
        experience_transforms,
        key,
        ExperienceNamedTuple(0 * jnp.ones((10, 5, 4)), 5 * jnp.ones((10, 5, 4))),
    )
    assert jnp.all(jnp.equal(experience.field_0, 1 * jnp.ones((10 * 5, 4))))
    assert jnp.all(jnp.equal(experience.field_1, 6 * jnp.ones((10 * 5, 4))))

    process_experience_pipeline = process_experience_pipeline_factory(
        False, True, ExperienceNamedTuple
    )
    experience = jax.jit(process_experience_pipeline)(
        experience_transforms,
        key,
        ExperienceNamedTuple(
            {"agent_0": 0 * jnp.ones((10, 4)), "agent_1": 0 * jnp.ones((10, 4))},
            {"agent_0": 5 * jnp.ones((10, 4)), "agent_1": 5 * jnp.ones((10, 4))},
        ),
    )
    assert jnp.all(jnp.equal(experience.field_0, 1 * jnp.ones((2 * 10, 4))))
    assert jnp.all(jnp.equal(experience.field_1, 6 * jnp.ones((2 * 10, 4))))

    process_experience_pipeline = process_experience_pipeline_factory(
        True, True, ExperienceNamedTuple
    )
    experience = jax.jit(process_experience_pipeline)(
        experience_transforms,
        key,
        ExperienceNamedTuple(
            {"agent_0": 0 * jnp.ones((10, 5, 4)), "agent_1": 0 * jnp.ones((10, 5, 4))},
            {"agent_0": 5 * jnp.ones((10, 5, 4)), "agent_1": 5 * jnp.ones((10, 5, 4))},
        ),
    )
    assert jnp.all(jnp.equal(experience.field_0, 1 * jnp.ones((2 * 5 * 10, 4))))
    assert jnp.all(jnp.equal(experience.field_1, 6 * jnp.ones((2 * 5 * 10, 4))))
