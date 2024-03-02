from typing import Callable

from flax.training.train_state import TrainState
from jrd_extensions import Seeded

from rl_tools.interface import IActor
from rl_tools.types import ActionType, ObsType, Array
from rl_tools.algos.factory import explore_general_factory


class PolicyActor(IActor, Seeded):
    """Wraps a policy_state into a deployed Actor."""

    def __init__(
        self,
        seed: int,
        policy_state: TrainState,
        select_action_fn: Callable,
        *,
        vectorized: bool = True,
    ) -> None:
        """Initializes a PolicyActor from a policy and a select_action function.

        Args:
            seed: An int for reproducibility.
            policy_state: A TrainState used in the select_action function.
            select_action_fn: A function that uses the policy_state to select an action.
        """
        Seeded.__init__(self, seed)
        self.policy_state = policy_state
        self.select_action_fn = explore_general_factory(
            select_action_fn, vectorized, False
        )

    def select_action(self, observation: ObsType) -> tuple[ActionType, Array]:
        return self.select_action_fn(self.policy_state, self.nextkey(), observation)
