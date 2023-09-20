from typing import SupportsFloat, Any, Optional

from gymnasium.core import WrapperActType, WrapperObsType

from wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


# Sound wrapper
class TerminateOnDeathWrapper(CleanUpFastResetWrapper):
    def __init__(self, env, **kwargs):
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
        return obs, info
