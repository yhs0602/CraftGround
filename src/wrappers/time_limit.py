from typing import SupportsFloat, Any, Optional

import gymnasium as gym
from gymnasium.core import WrapperActType, WrapperObsType


class TimeLimitWrapper(gym.Wrapper):
    def __init__(self, env, max_timesteps: int, **kwargs):
        self.env = env
        self.max_timesteps = max_timesteps
        super().__init__(self.env)
        self.timestep = 0

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.timestep += 1
        if self.timestep >= self.max_timesteps:
            terminated = True

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )  # , done: deprecated

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.timestep = 0
        return obs, info
