from collections import deque
from typing import SupportsFloat, Any, Optional

from gymnasium.core import WrapperActType, WrapperObsType

from final_experiments.wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


# Go up wrapper
class GoUpWrapper(CleanUpFastResetWrapper):
    def __init__(self, env, target_height):
        self.env = env
        self.target_height = target_height
        super().__init__(self.env)
        self.height_deque = deque(maxlen=2)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info_obs = info["obs"]

        self.height_deque.append(info_obs.y)

        if self.height_deque[0] < self.height_deque[1]:
            reward = 0.1
            print("Got higher!")
        elif self.height_deque[0] > self.height_deque[1]:
            reward = -0.1
            print("Got lower!")

        near_campfire = False
        if info_obs.sound_subtitles:
            for sound in info_obs.sound_subtitles:
                if sound.translate_key == "subtitles.block.campfire.crackle":
                    near_campfire = True

        if near_campfire:
            reward += 0.002  # guide toward campfire
        else:
            reward -= 0.001  # time penalty

        if info_obs.z > -33:
            reward = -0.01  # went out of bounds

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

    def reset(
        self,
        fast_reset: bool = True,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options, fast_reset=fast_reset)
        info_obs = info["obs"]
        self.height_deque.clear()
        self.height_deque.append(info_obs.y)
        return obs, info
