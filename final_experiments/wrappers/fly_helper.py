from typing import SupportsFloat, Any, Optional, Tuple

from gymnasium.core import WrapperActType, WrapperObsType

from final_experiments.wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


# Sound wrapper
class FlyHelperWrapper(CleanUpFastResetWrapper):
    def __init__(self, env):
        self.env = env
        self.prev_flight_distance = None
        self.prev_firework_stock = 64
        super().__init__(self.env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info_obs = info["obs"]
        flight_distance = info_obs.misc_statistics["aviate_one_cm"]
        inventory = info_obs.inventory
        firework_stock = 0
        for item in inventory:
            if item.translation_key == "item.minecraft.firework_rocket":
                firework_stock = item.count
        if firework_stock < self.prev_firework_stock:
            reward -= 0.05  # penalty for using firework rocket
        self.prev_firework_stock = firework_stock

        if flight_distance > self.prev_flight_distance:
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
        self.prev_firework_stock = 64
        return obs, info
