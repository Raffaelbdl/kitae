import gymnasium as gym


class DummyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    action_space = gym.spaces.Box(-1, 1, ())
    observation_space = gym.spaces.Box(-1, 1, ())

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.render_mode = render_mode
