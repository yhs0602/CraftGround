from collections import deque
from typing import SupportsFloat, Any, Optional

from gymnasium.core import WrapperActType, WrapperObsType

from final_experiments.wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


# Sound wrapper
class AvoidDamageWrapper(CleanUpFastResetWrapper):
    def __init__(self, env):
        self.env = env
        super().__init__(self.env)
        self.health_deque = deque(maxlen=2)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info_obs = info["obs"]
        is_dead = info_obs.is_dead

        reward = 0.5
        if is_dead:
            self.health_deque.append(20)
            terminated = True
            reward = -1

        else:
            self.health_deque.append(info_obs.health)
            if self.health_deque[0] < self.health_deque[1]:
                reward = -0.1

        return (
            obs,
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
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options, fast_reset=fast_reset)
        info_obs = info["obs"]
        self.health_deque.clear()
        self.health_deque.append(info_obs.health)
        return obs, info
