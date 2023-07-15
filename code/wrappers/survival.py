from typing import SupportsFloat, Any

from gymnasium.core import WrapperActType, WrapperObsType

from code.wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


# Sound wrapper
class SurvivalWrapper(CleanUpFastResetWrapper):
    def __init__(self, env):
        self.env = env
        super().__init__(self.env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info_obs = info["obs"]
        is_dead = info_obs.is_dead

        if is_dead:
            terminated = True
            reward = -1
        else:
            reward = 0.5

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )  # , done: deprecated
