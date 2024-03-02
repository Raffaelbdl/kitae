from flax.training.train_state import TrainState
import numpy as np
from optax import sgd

from rl_tools.actor import PolicyActor


def test_policy_actor():
    policy_state = TrainState.create(
        apply_fn=lambda p, x: x + 1, params={}, tx=sgd(0.0)
    )
    select_action_fn = lambda s, k, o: s.apply_fn(s.params, o)

    policy_actor = PolicyActor(
        0,
        policy_state=policy_state,
        select_action_fn=select_action_fn,
        vectorized=False,
    )
    assert np.array_equal(policy_actor.select_action(np.zeros(())), np.ones(()))

    policy_actor = PolicyActor(
        0,
        policy_state=policy_state,
        select_action_fn=select_action_fn,
        vectorized=True,
    )
    assert np.array_equal(policy_actor.select_action(np.zeros((1, 5))), np.ones((1, 5)))


test_policy_actor()
