# import pdb
from typing import Dict, Any


# initial_env = InitialEnvironment(["sword", "shield"], [10, 20], ["summon ", "killMob"], 800, 600, 123456, True, False, False, "sunny")
class InitialEnvironment:
    def __init__(
        self,
        initialInventoryCommands,
        initialPosition,
        initialMobsCommands,
        imageSizeX,
        imageSizeY,
        seed,
        allowMobSpawn,
        alwaysNight,
        alwaysDay,
        initialWeather,
    ):
        self.initialInventoryCommands = initialInventoryCommands
        self.initialPosition = initialPosition
        self.initialMobsCommands = initialMobsCommands
        self.imageSizeX = imageSizeX
        self.imageSizeY = imageSizeY
        self.seed = seed
        self.allowMobSpawn = allowMobSpawn
        self.alwaysNight = alwaysNight
        self.alwaysDay = alwaysDay
        self.initialWeather = initialWeather

    def to_dict(self) -> Dict[str, Any]:
        initial_env_dict = {
            "initialInventoryCommands": self.initialInventoryCommands,
            "initialPosition": self.initialPosition,
            "initialMobsCommands": self.initialMobsCommands,
            "imageSizeX": self.imageSizeX,
            "imageSizeY": self.imageSizeY,
            "seed": self.seed,
            "allowMobSpawn": self.allowMobSpawn,
            "alwaysNight": self.alwaysNight,
            "alwaysDay": self.alwaysDay,
            "initialWeather": self.initialWeather,
        }
        return {k: v for k, v in initial_env_dict.items() if v is not None}
