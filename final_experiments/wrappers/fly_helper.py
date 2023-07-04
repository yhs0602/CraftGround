from typing import SupportsFloat, Any, Optional, Tuple

from gymnasium.core import WrapperActType, WrapperObsType

from final_experiments.wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


# Sound wrapper
class FlyHelperWrapper(CleanUpFastResetWrapper):
    def __init__(self, env):
        self.env = env
        self.prev_flight_distance = None
        super().__init__(self.env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if action == 0:
            reward += 0.001
        info_obs = info["obs"]
        flight_distance = info_obs.misc_statistics["aviate_one_cm"]
        if flight_distance > self.prev_flight_distance:
            print("Flew")
            reward += 0.01
        self.prev_flight_distance = flight_distance

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
    ) -> Tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(fast_reset=True, seed=seed, options=options)
        info_obs = info["obs"]
        self.prev_flight_distance = info_obs.misc_statistics["aviate_one_cm"]
        return obs, info
