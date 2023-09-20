import mydojo
from environments.base_environment import BaseEnvironment


class HusksEnvironment(BaseEnvironment):
    def make(
        self,
        verbose: bool,
        env_path: str,
        port: int,
        size_x: int = 114,
        size_y: int = 64,
        hud: bool = False,
        render_action: bool = True,
        *args,
        **kwargs,
    ):
        return mydojo.make(
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
                "minecraft:husk ~ ~ ~15 {HandItems:[{Count:1,id:iron_shovel},{}]}",
                "minecraft:husk ~-15 ~ ~15 {HandItems:[{Count:1,id:iron_shovel},{}]}",
                "minecraft:husk ~-15 ~ ~ {HandItems:[{Count:1,id:iron_shovel},{}]}",
                "minecraft:husk ~15 ~ ~ {HandItems:[{Count:1,id:iron_shovel},{}]}",
                "minecraft:husk ~ ~ ~-15 {HandItems:[{Count:1,id:iron_shovel},{}]}",
                # player looks at south (positive Z) when spawn
            ],
            imageSizeX=size_x,
            imageSizeY=size_y,
            visibleSizeX=size_x,
            visibleSizeY=size_y,
            seed=12345,  # nullable
            allowMobSpawn=False,
            alwaysDay=True,
            alwaysNight=False,
            initialWeather="clear",  # nullable
            isHardCore=False,
            isWorldFlat=True,  # superflat world
            obs_keys=["sound_subtitles"],
            isHudHidden=not hud,
            render_action=render_action,
        ), [
            "subtitles.entity.husk.ambient",
            "subtitles.block.generic.footsteps",
        ]
