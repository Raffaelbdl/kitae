from dataclasses import asdict, dataclass
from pathlib import Path

import cloudpickle
from gymnasium.spaces import Space
import yaml


@dataclass
class AlgoParams: ...


@dataclass
class UpdateConfig:
    learning_rate: float
    learning_rate_annealing: bool
    max_grad_norm: float
    max_buffer_size: int
    batch_size: int
    n_epochs: int
    shared_encoder: bool


@dataclass
class TrainConfig:
    n_env_steps: int
    save_frequency: int


@dataclass
class EnvConfig:
    task_name: str
    observation_space: Space
    action_space: Space
    n_envs: int
    n_agents: int


@dataclass
class AlgoConfig:
    seed: int
    algo_params: AlgoParams
    update_cfg: UpdateConfig
    train_cfg: TrainConfig
    env_cfg: EnvConfig


from save.serializable import Serializable


class ConfigSerializable(Serializable):
    """Static Serializable class for AlgoConfig."""

    @staticmethod
    def serialize(config: AlgoConfig, path: Path):
        env_cfg = config.env_cfg

        # save custom instance type for yaml-dumping
        algo_params_type = type(config.algo_params)
        with path.joinpath("algo_params_type").open("wb") as f:
            cloudpickle.dump(algo_params_type, f)

        algo_config_dict = asdict(config)
        algo_config_dict.pop("env_cfg")  # pop because cannot be yaml-dumped

        with path.joinpath("algo_config.yaml").open("w") as f:
            yaml.dump(algo_config_dict, f)
        with path.joinpath("env_config").open("wb") as f:
            cloudpickle.dump(env_cfg, f)

    @staticmethod
    def unserialize(path: Path) -> AlgoConfig:

        with path.joinpath("algo_params_type").open("rb") as f:
            algo_params_type = cloudpickle.load(f)

        with path.joinpath("algo_config.yaml").open("r") as f:
            algo_config_dict = yaml.load(f, yaml.SafeLoader)
        with path.joinpath("env_config").open("rb") as f:
            algo_config_dict["env_cfg"] = cloudpickle.load(f)

        return AlgoConfig(
            seed=algo_config_dict["seed"],
            algo_params=algo_params_type(**algo_config_dict["algo_params"]),
            update_cfg=UpdateConfig(**algo_config_dict["update_cfg"]),
            train_cfg=TrainConfig(**algo_config_dict["train_cfg"]),
            env_cfg=algo_config_dict["env_cfg"],
        )
