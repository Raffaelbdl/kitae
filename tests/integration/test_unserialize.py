from typing import Callable
from rl_tools.base import BaseAgent
from rl_tools.buffer import Experience
from rl_tools.config import AlgoConfig
from rl_tools.interface import IBuffer


def dummy_config():
    from dataclasses import dataclass
    import gymnasium as gym
    from rl_tools.config import (
        AlgoConfig,
        AlgoParams,
        UpdateConfig,
        TrainConfig,
        EnvConfig,
    )

    @dataclass
    class CustomParams(AlgoParams):
        A: int
        B: str

    ap = CustomParams(A=0, B="abc")
    uc = UpdateConfig(
        learning_rate=1e-4,
        learning_rate_annealing=False,
        max_grad_norm=0.5,
        max_buffer_size=256,
        batch_size=32,
        n_epochs=1,
        shared_encoder=False,
    )
    tc = TrainConfig(int(1e6), -1)
    ec = EnvConfig("Task-v2", gym.spaces.Discrete(5), gym.spaces.Box(-1, -1, ()), 5, 1)

    return AlgoConfig(seed=0, algo_params=ap, update_cfg=uc, train_cfg=tc, env_cfg=ec)


def dummy_train_state_factory(*args, **kwargs):
    return lambda *args, **kwargs: None


def dummy_explore_factory(*args, **kwargs):
    return lambda *args, **kwargs: None, None


def dummy_process_experience_factory(*args, **kwargs):
    return lambda *args, **kwargs: None


def dummy_update_step_factory(*args, **kwargs):
    return lambda *args, **kwargs: None


class DummyAgent(BaseAgent):
    def __init__(
        self,
        run_name: str,
        config: AlgoConfig,
        *,
        preprocess_fn: Callable = None,
        tabulate: bool = False,
        experience_type: bool = ...,
    ):
        super().__init__(
            run_name,
            config,
            dummy_train_state_factory,
            dummy_explore_factory,
            dummy_process_experience_factory,
            dummy_update_step_factory,
            preprocess_fn=preprocess_fn,
            tabulate=tabulate,
            experience_type=experience_type,
        )

    def should_update(self, step: int, buffer: IBuffer) -> bool:
        return False


def test_unserialize():
    agent = DummyAgent("tmp_run_name", dummy_config())
    new_agent = DummyAgent.unserialize("./results/tmp_run_name/")
    assert True


if __name__ == "__main__":
    test_unserialize()
