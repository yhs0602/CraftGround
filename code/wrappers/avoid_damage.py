from collections import deque
from typing import SupportsFloat, Any, Optional

from gymnasium.core import WrapperActType, WrapperObsType

from wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


# Sound wrapper
class AvoidDamageWrapper(CleanUpFastResetWrapper):
    def __init__(
        self, env, alive_reward=0.5, damage_reward=-0.1, death_reward=-1, **kwargs
    ):
        self.env = env
        self.base_reward = alive_reward
        self.damage_reward = damage_reward
        self.death_reward = death_reward
        super().__init__(self.env)
        self.health_deque = deque(maxlen=2)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info_obs = info["obs"]
        is_dead = info_obs.is_dead

        reward = self.base_reward
        if is_dead:
            self.health_deque.append(20)
            terminated = True
            reward = self.death_reward

        else:
            self.health_deque.append(info_obs.health)
            if self.health_deque[0] < self.health_deque[1]:
                reward = self.damage_reward

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
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options, fast_reset=fast_reset)
        info_obs = info["obs"]
        self.health_deque.clear()
        self.health_deque.append(info_obs.health)
        return obs, info
