from typing import Any

from gymnasium import Wrapper


class EnvpoolCompatibility(Wrapper):
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        return self.env.reset()


class SubProcVecParallelEnvCompatibility(Wrapper):
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        return self.env.reset()
