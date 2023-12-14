from dataclasses import asdict, dataclass

from gymnasium.spaces import Space
from ml_collections import ConfigDict


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
        self.train_cfg.n_envs = self.env_cfg.n_envs
        self.train_cfg.n_agents = self.env_cfg.n_agents
