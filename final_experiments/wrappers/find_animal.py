from typing import SupportsFloat, Any, Optional

import gymnasium as gym
from gymnasium.core import WrapperActType, WrapperObsType


# Sound wrapper
class FindAnimalWrapper(gym.Wrapper):
    def __init__(self, env, target_translation_key: str, target_number):
        self.env = env
        self.target_translation_key = target_translation_key
        self.target_number = target_number
        super().__init__(self.env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info_obs = info["obs"]
        surrounding_entities = info_obs.surrounding_entities

        animals1 = self.count_animals(surrounding_entities[1].entities)
        animals5 = self.count_animals(surrounding_entities[5].entities)
        animals10 = self.count_animals(surrounding_entities[10].entities)

        if animals1 >= self.target_number:
            reward = 1
            terminated = True
        else:
            reward = (
                0.5 * animals1 / self.target_number
                + 0.2 * animals5 / self.target_number
                + 0.1 * animals10 / self.target_number
                - 0.81
            )

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

    def count_animals(self, animal_list) -> int:
        count = 0
        for animal in animal_list:
            if animal.translation_key == self.target_translation_key:
                count += 1
        return count
