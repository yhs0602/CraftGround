import gymnasium as gym
import numpy as np

import mydojo
from env_wrappers.husk_sound_wrapper import HuskSoundWrapper


class HusksSoundWrapper(HuskSoundWrapper):
    def __init__(self, verbose=False, env_path=None, port=8000):
        self.env = mydojo.make(
            verbose=verbose,
            env_path=env_path,
            port=port,
            initialInventoryCommands=[],
            initialPosition=None,  # nullable
            initialMobsCommands=[
                # "minecraft:sheep",
                "minecraft:husk ~ ~ ~5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
                "minecraft:husk ~5 ~ ~5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
                "minecraft:husk ~-5 ~ ~-5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
                # player looks at south (positive Z) when spawn
            ],
            imageSizeX=114,
            imageSizeY=64,
            visibleSizeX=114,
            visibleSizeY=64,
            seed=12345,  # nullable
            allowMobSpawn=False,
            alwaysDay=True,
            alwaysNight=False,
            initialWeather="clear",  # nullable
            isHardCore=False,
            isWorldFlat=True,  # superflat world
            obs_keys=["sound_subtitles"],
        )
        super().__init__(self.env)
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(7,), dtype=np.float32
        )
