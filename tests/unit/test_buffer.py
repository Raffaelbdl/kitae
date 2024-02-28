import pytest

import jax
import jax.numpy as jnp

from rl_tools.buffer import stack_experiences, Experience
from rl_tools.buffer import batchify
from rl_tools.buffer import batchify_and_randomize


def test_stack_experiences():
    experiences = [
        Experience(
            observation=jnp.zeros((5,)),
            action=jnp.zeros(()),
            reward=jnp.zeros(()),
            done=jnp.zeros(()),
            next_observation=jnp.zeros((5,)),
            log_prob=jnp.zeros(()),
        ),
        Experience(
            observation=jnp.ones((5,)),
            action=jnp.ones(()),
            reward=jnp.ones(()),
            done=jnp.ones(()),
            next_observation=jnp.ones((5,)),
            log_prob=jnp.ones(()),
        ),
        Experience(
            observation=2 * jnp.ones((5,)),
            action=2 * jnp.ones(()),
            reward=2 * jnp.ones(()),
            done=2 * jnp.ones(()),
            next_observation=2 * jnp.ones((5,)),
            log_prob=2 * jnp.ones(()),
        ),
    ]
    stacked = stack_experiences(experiences)

    assert stacked.observation.shape == (3, 5)
    assert stacked.done.shape == (3,)

    assert jnp.equal(
        stacked.observation,
        jnp.stack([jnp.zeros((5,)), jnp.ones((5,)), 2 * jnp.ones((5,))], axis=0),
    ).all()
    assert jnp.equal(
        stacked.action,
        jnp.stack([jnp.zeros(()), jnp.ones(()), 2 * jnp.ones(())], axis=0),
    ).all()

    experiences = [
        Experience(
            observation=jnp.zeros((4, 5)),
            action=jnp.zeros((4,)),
            reward=jnp.zeros((4,)),
            done=jnp.zeros((4,)),
            next_observation=jnp.zeros((4, 5)),
            log_prob=jnp.zeros((4,)),
        ),
        Experience(
            observation=jnp.ones((4, 5)),
            action=jnp.ones((4,)),
            reward=jnp.ones((4,)),
            done=jnp.ones((4,)),
            next_observation=jnp.ones((4, 5)),
            log_prob=jnp.ones((4,)),
        ),
        Experience(
            observation=2 * jnp.ones((4, 5)),
            action=2 * jnp.ones((4,)),
            reward=2 * jnp.ones((4,)),
            done=2 * jnp.ones((4,)),
            next_observation=2 * jnp.ones((4, 5)),
            log_prob=2 * jnp.ones((4,)),
        ),
    ]
    stacked = stack_experiences(experiences)

    assert stacked.observation.shape == (3, 4, 5)
    assert stacked.done.shape == (3, 4)

    assert jnp.equal(
        stacked.observation,
        jnp.stack([jnp.zeros((4, 5)), jnp.ones((4, 5)), 2 * jnp.ones((4, 5))], axis=0),
    ).all()
    assert jnp.equal(
        stacked.action,
        jnp.stack([jnp.zeros((4,)), jnp.ones((4,)), 2 * jnp.ones((4,))], axis=0),
    ).all()


def test_batchify():
    data = (jnp.zeros((5,)), jnp.ones((5, 10)))

    batches = batchify(data, 2)
    assert batches[0].shape == (2, 2)
    assert batches[1].shape == (2, 2, 10)

    batches = batchify(data, 3)
    assert batches[0].shape == (1, 3)
    assert batches[1].shape == (1, 3, 10)

    with pytest.raises(AssertionError):
        batches = batchify(data, 6)


def test_batchify_and_randomize():
    key = jax.random.key(0)
    data = (jnp.zeros((5,)), jnp.ones((5, 10)))

    batches = batchify_and_randomize(key, data, 2)
    assert batches[0].shape == (2, 2)
    assert batches[1].shape == (2, 2, 10)

    batches = batchify_and_randomize(key, data, 3)
    assert batches[0].shape == (1, 3)
    assert batches[1].shape == (1, 3, 10)

    with pytest.raises(AssertionError):
        batches = batchify_and_randomize(key, data, 6)

    data = (jnp.zeros((100,)), jnp.ones((100, 10)))
    batches_0 = batchify_and_randomize(key, data, 2)
    batches_1 = batchify_and_randomize(key, data, 2)

    assert jnp.array_equal(batches_0[0], batches_1[0])
    assert jnp.array_equal(batches_0[1], batches_1[1])

    key0, key1 = jax.random.split(key, 2)
    data = (jnp.arange(5).reshape((5,)), jnp.arange(5 * 10).reshape((5, 10)))
    batches_0 = batchify_and_randomize(key0, data, 2)
    batches_1 = batchify_and_randomize(key1, data, 2)

    assert jnp.any(jnp.not_equal(batches_0[0], batches_1[0]))
    assert jnp.any(jnp.not_equal(batches_0[1], batches_1[1]))
