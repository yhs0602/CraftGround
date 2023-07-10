from collections import deque
from typing import SupportsFloat, Any, Optional

from gymnasium.core import WrapperActType, WrapperObsType

from final_experiments.wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


# Sound wrapper
class AttackKillWrapper(CleanUpFastResetWrapper):
    def __init__(self, env):
        self.env = env
        self.durabilities = deque(maxlen=2)
        self.old_killed_stat = 0
        super().__init__(self.env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info_obs = info["obs"]
        main_hand_item = info_obs.inventory[0]
        self.durabilities.append(main_hand_item.durability)
        if (
            self.durabilities[0] > self.durabilities[1]
        ):  # durability decreased. We attacked successfully
            reward -= 0.1  # Make it kill with fewer attacks
        print(f"{info_obs.killed_statistics=}")
        if info_obs.killed_statistics["husk"] > self.old_killed_stat:
            reward += 1  # we killed a husk
            self.old_killed_stat = info_obs.killed_statistics["husk"]
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
        fast_reset: bool = True,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(fast_reset=fast_reset, seed=seed, options=options)
        info_obs = info["obs"]
        self.old_killed_stat = info_obs.killed_statistics["husk"]
        self.durabilities.clear()
        self.durabilities.append(info_obs.inventory[0].durability)
        return obs, info
