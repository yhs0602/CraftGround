from typing import SupportsFloat, Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType

from wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


# Sound wrapper
class VisionWrapper(CleanUpFastResetWrapper):
    def __init__(self, env, x_dim, y_dim, **kwargs):
        self.env = env
        super().__init__(self.env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, x_dim, y_dim),
            dtype=np.uint8,
        )

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        rgb = info["rgb"]
        return (
            rgb,
            reward,
            terminated,
            truncated,
            info,
        )  # , done: deprecated

    def reset(
        self,
        fast_reset: bool = True,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None
    ):
        obs, info = self.env.reset(fast_reset=fast_reset, seed=seed, options=options)
        rgb = info["rgb"]
        return rgb, info
