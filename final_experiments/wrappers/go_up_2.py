from typing import SupportsFloat, Any

from gymnasium.core import WrapperActType, WrapperObsType

from final_experiments.wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


# Go up wrapper
class GoUp2Wrapper(CleanUpFastResetWrapper):
    def __init__(self, env, target_height):
        self.env = env
        self.target_height = target_height
        super().__init__(self.env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info_obs = info["obs"]

        reward = (self.target_height - info_obs.y) * -0.01

        if info_obs.z > -33:
            reward -= 0.01  # went out of bounds

        if info_obs.y >= self.target_height:
            reward = 1
            terminated = True

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )  # , done: deprecated
