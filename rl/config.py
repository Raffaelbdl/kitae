from gymnasium.spaces import Space


class EnvConfig:
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        n_envs: int,
        n_agents: int,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_envs = n_envs
        self.n_agents = n_agents
