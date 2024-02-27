import jax.numpy as jnp

from rl_tools.buffer import stack_experiences, Experience


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
