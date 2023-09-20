from typing import SupportsFloat, Any, Optional

from gymnasium.core import WrapperActType, WrapperObsType

from wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


# Sound wrapper
class FishTokenWrapper(CleanUpFastResetWrapper):
    def __init__(self, env, **kwargs):
        self.env = env
        self.gave_reward = False
        super().__init__(self.env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info_obs = info["obs"]
        experience = info_obs.misc_statistics["experience"]
        # print(fish_caught)
        self.experience_deque.append(experience)
        if len(self.experience_deque) == 2:
            if (
                self.experience_deque[1] > self.experience_deque[0]
            ):  # fish_caught increased
                # print("Fish Caught")
                reward += 1
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
        self.experience_deque.clear()
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info
