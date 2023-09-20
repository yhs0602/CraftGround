from typing import Any, Optional

import gymnasium as gym
from gymnasium.core import WrapperObsType


class CleanUpFastResetWrapper(gym.Wrapper):
    def terminate(self):
        if hasattr(self.env, "terminate"):
            self.env.terminate()
        else:
            print("Warning: terminate() not implemented in environment.")

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info
