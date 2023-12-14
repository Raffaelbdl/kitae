from dataclasses import asdict, dataclass

from gymnasium.spaces import Space
from ml_collections import ConfigDict
import yaml


@dataclass
class AlgoParams:
    ...


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


class AlgoConfig(ConfigDict):
    """Base configuration class for all algorithms

    Contains all information about an instance of an agent.
    """

    def __init__(
        self,
        seed: int,
        algo_params: AlgoParams,
        update_config: UpdateConfig,
        train_config: TrainConfig,
        env_config: EnvConfig,
    ):
        """Creates an instance of AlgoConfig.

        Args:
            seed: An int that enforces code reproducibility.
            algo_params: An algorithm-specific instance of AlgoParams.
            update_config: An instance of UpdateConfig.
            train_config: An instance of TrainConfig.
            env_config: An instance of EnvConfig.
        """
        config = {
            "seed": seed,
            "algo_params": asdict(algo_params),
            "update_cfg": asdict(update_config),
            "train_cfg": asdict(train_config),
            "env_cfg": asdict(env_config),
        }
        ConfigDict.__init__(self, config)

        # env_cfg cannot be serialized to yaml because it holds gymnasium spaces.
        # The chosen solution is to transfer the meaningful information for the user.
        self.task_name = self.env_cfg.task_name
        self.train_cfg.n_envs = self.env_cfg.n_envs
        self.train_cfg.n_agents = self.env_cfg.n_agents

    @classmethod
    def create_config_from_file(
        cls, config_path: str, env_config: EnvConfig
    ) -> ConfigDict:
        """Creates an AlgoConfig instance from a yaml file.

        Technically, the created instance is not from the AlgoConfig class. But
        it does behave exactly the same.

        Args:
            config_path: A string path to the yaml configuration file.
            env_config: An instance of EnvConfig.

        Returns:
            An instance of ConfigDict that behaves similarly to AlgoConfig.

        Raises:
            KeyError: If one key is missing from the yaml config file.
        """
        with open(config_path, "r") as file:
            config_dict = yaml.load(file, yaml.SafeLoader)

        for key in ["seed", "algo_params", "update_cfg", "train_cfg"]:
            if key not in config_dict.keys():
                raise KeyError(
                    f"KeyError : {key} key is not present in the config file"
                )

        config = ConfigDict(config_dict | {"env_cfg": asdict(env_config)})

        config.task_name = config.env_cfg.task_name
        config.train_cfg.n_envs = config.env_cfg.n_envs
        config.train_cfg.n_agents = config.env_cfg.n_agents

        return config
