from typing import SupportsFloat, Any, Optional

from gymnasium.core import WrapperActType, WrapperObsType

from final_experiments.wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


# Sound wrapper
class FindVillageWrapper(CleanUpFastResetWrapper):
    def __init__(self, env):
        self.env = env
        super().__init__(self.env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info_obs = info["obs"]
        surrounding_entities = info_obs.surrounding_entities

        villager30 = self.count_villagers(surrounding_entities[30].entities)

        if villager30 >= 1:
            reward = 5
            terminated = True
        elif info_obs.is_dead:
            reward = -1
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
        return obs, info

    def count_villagers(self, entity_list) -> int:
        count = 0
        for entity in entity_list:
            if entity.translation_key == "entity.minecraft.villager":
                count += 1
        return count
