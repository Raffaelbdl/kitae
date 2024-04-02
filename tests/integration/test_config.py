from rl_tools.config import AlgoConfig, AlgoParams, UpdateConfig, TrainConfig, EnvConfig
from rl_tools.config import dump_algo_config, load_algo_config

import time


def test_dump_load():
    import shutil
    import gymnasium as gym

    ap = AlgoParams()
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

    algo_config = AlgoConfig(
        seed=0, algo_params=ap, update_cfg=uc, train_cfg=tc, env_cfg=ec
    )
    dump_algo_config(algo_config, "./tmp_cfg_test/")
    new_config = load_algo_config("./tmp_cfg_test/")

    assert algo_config == new_config
    shutil.rmtree("./tmp_cfg_test/")


def test_dump_load_custom_params():
    from dataclasses import dataclass
    import shutil
    import gymnasium as gym

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

    algo_config = AlgoConfig(
        seed=0, algo_params=ap, update_cfg=uc, train_cfg=tc, env_cfg=ec
    )
    dump_algo_config(algo_config, "./tmp_cfg_test/")
    new_config = load_algo_config("./tmp_cfg_test/")

    assert algo_config == new_config
    shutil.rmtree("./tmp_cfg_test/")
