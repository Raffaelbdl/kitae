import shutil
from typing import Callable

from kitae.agent import BaseAgent
from kitae.config import AlgoConfig, UpdateConfig, TrainConfig, EnvConfig
from kitae.interface import IBuffer


def dummy_config() -> AlgoConfig:
    from dataclasses import dataclass
    import gymnasium as gym
    from kitae.config import (
        AlgoConfig,
        AlgoParams,
        UpdateConfig,
        TrainConfig,
        EnvConfig,
    )

    @dataclass
    class CustomParams:
        A: int = 0
        B: str = "foo"

    ap = CustomParams(A=1, B="bar")
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


def dummy_one_output_factory(*args, **kwargs) -> Callable:
    return lambda *args, **kwargs: None


def dummy_two_output_factory(*args, **kwargs) -> Callable:
    return lambda *args, **kwargs: None, None


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
            dummy_one_output_factory,
            dummy_two_output_factory,
            dummy_one_output_factory,
            dummy_one_output_factory,
            preprocess_fn=preprocess_fn,
            tabulate=tabulate,
            experience_type=experience_type,
        )

    def should_update(self, step: int, buffer: IBuffer) -> bool:
        return False


def test_unserialize():
    agent = DummyAgent("tmp_run_name", dummy_config(), preprocess_fn=lambda x: x + 1)
    unserialized = DummyAgent.unserialize("./runs/tmp_run_name")
    assert True

    shutil.rmtree("./runs/tmp_run_name")

    assert agent.run_name == unserialized.run_name
    assert agent.config == unserialized.config
    assert agent.preprocess_fn(10) == unserialized.preprocess_fn(10)
