import craftground
from environments.base_environment import BaseEnvironment


class MansionEnvironment(BaseEnvironment):
    def make(
        self,
        verbose: bool,
        env_path: str,
        port: int,
        size_x: int = 114,
        size_y: int = 64,
        hud: bool = False,
        render_action: bool = True,
        render_distance: int = 2,
        simulation_distance: int = 5,
        *args,
        **kwargs,
    ):
        build_mansion = [
            "difficulty peaceful",  # peaceful mode
            "place structure mansion -26 80 -40",  # place a mansion
            "tp @p -32 78 -35 -180 0",  # tp player to the mansion's start point
            "setblock -22 86 -51 campfire",
            "setblock -21 86 -51 campfire",
            "setblock -23 86 -51 campfire",  # beacons
            "effect give @p night_vision infinite 1 true",  # night vision without particles
            "kill @e[type=!player]",  # kill all mobs, items except player
        ]

        return craftground.make(
            verbose=verbose,
            env_path=env_path,
            port=port,
            initialInventoryCommands=[],
            initialPosition=[-32, 78, -35],  # nullable
            initialMobsCommands=[],
            imageSizeX=size_x,
            imageSizeY=size_y,
            visibleSizeX=size_x,
            visibleSizeY=size_y,
            seed=8952232712572833477,  # nullable
            allowMobSpawn=False,
            alwaysDay=True,
            alwaysNight=False,
            initialWeather="clear",  # nullable
            isHardCore=False,
            isWorldFlat=False,  # superflat world
            obs_keys=["sound_subtitles"],
            initialExtraCommands=build_mansion,
            isHudHidden=not hud,
            render_action=render_action,
            render_distance=render_distance,
            simulation_distance=simulation_distance,
        ), [
            "subtitles.block.campfire.crackle",  # Campfire crackles
        ]
