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
        isHardCore,
        isWorldFlat,
        visibleSizeX=None,
        visibleSizeY=None,
        initialExtraCommands=None,
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
        self.isHardCore = isHardCore
        self.isWorldFlat = isWorldFlat
        self.visibleSizeX = imageSizeX if visibleSizeX is None else visibleSizeX
        self.visibleSizeY = imageSizeY if visibleSizeY is None else visibleSizeY
        self.initialExtraCommands = initialExtraCommands

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
            "isHardCore": self.isHardCore,  # ignored for now
            "isWorldFlat": self.isWorldFlat,
            "visibleSizeX": self.visibleSizeX,
            "visibleSizeY": self.visibleSizeY,
            "initialExtraCommands": self.initialExtraCommands,
        }
        return {k: v for k, v in initial_env_dict.items() if v is not None}
