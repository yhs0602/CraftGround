import random
from typing import Optional, Any

from gymnasium.core import WrapperObsType
from craftground import make
from environments.base_environment import BaseEnvironment
from wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


class FindAnimalEnvironment(BaseEnvironment):
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
        build_cage_comands = [
            "tp @p 0 -59 0",  # tp player
            "fill ~-11 ~-1 ~-11 ~11 ~2 ~11 minecraft:hay_block hollow",  # make a cage
            "fill ~-10 ~-1 ~-10 ~-7 ~-1 ~-7 minecraft:acacia_fence outline",  # make a cage
            "fill ~7 ~-1 ~7 ~10 ~-1 ~10 minecraft:acacia_fence outline",  # make a cage
            "fill ~-10 ~-1 ~7 ~-7 ~-1 ~10 minecraft:acacia_fence outline",  # make a cage
            "fill ~7 ~-1 ~-10 ~10 ~-1 ~-7 minecraft:acacia_fence outline",  # make a cage
            "fill ~-9 ~-1 ~-9 ~-8 ~-1 ~-8 minecraft:air outline",  # make a cage
            "fill ~8 ~-1 ~8 ~9 ~-1 ~9 minecraft:air outline",  # make a cage
            "fill ~-9 ~-1 ~8 ~-8 ~-1 ~9 minecraft:air outline",  # make a cage
            "fill ~8 ~-1 ~-9 ~9 ~-1 ~-8 minecraft:air outline",  # make a cage
            "fill ~-11 ~2 ~-11 ~11 ~10 ~11 minecraft:air replace",  # make a cage
        ]

        def summon_animal_commands(animal, x, z):
            return f"summon minecraft:{animal} ~{x} ~-1 ~{z}"

        coords = [
            (9, 9),
            (9, -9),
            (-9, 9),
            (-9, -9),
        ]

        class RandomAnimalWrapper(CleanUpFastResetWrapper):
            def __init__(self):
                random.shuffle(coords)
                summon_animal_commands_list = [
                    summon_animal_commands("sheep", coords[0][0], coords[0][1]),
                    summon_animal_commands("pig", coords[1][0], coords[1][1]),
                    summon_animal_commands("chicken", coords[2][0], coords[2][1]),
                ] * 7
                self.env = make(
                    verbose=verbose,
                    env_path=env_path,
                    port=port,
                    initialInventoryCommands=[],
                    initialPosition=None,  # nullable
                    initialMobsCommands=[],
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
                    initialExtraCommands=build_cage_comands
                    + summon_animal_commands_list,
                    surrounding_entities_keys=[1, 2, 5],
                    isHudHidden=not hud,
                    render_action=render_action,
                    render_distance=render_distance,
                    simulation_distance=simulation_distance,
                )
                super(RandomAnimalWrapper, self).__init__(self.env)

            def reset(
                self,
                *,
                seed: Optional[int] = None,
                options: Optional[dict[str, Any]] = None,
            ) -> tuple[WrapperObsType, dict[str, Any]]:
                extra_commands = ["tp @e[type=!player] ~ -500 ~"]
                random.shuffle(coords)
                summon_animal_commands_list = [
                    summon_animal_commands("sheep", coords[0][0], coords[0][1]),
                    summon_animal_commands("pig", coords[1][0], coords[1][1]),
                    summon_animal_commands("chicken", coords[2][0], coords[2][1]),
                ] * 7
                extra_commands.extend(summon_animal_commands_list)

                obs = self.env.reset(
                    seed=seed,
                    options={"fast_reset": True, "extra_commands": extra_commands},
                )
                return obs

        return RandomAnimalWrapper(), [
            "subtitles.entity.sheep.ambient",  # sheep ambient sound
            "subtitles.block.generic.footsteps",  # player, animal walking
            "subtitles.block.generic.break",  # sheep eating grass
            "subtitles.entity.cow.ambient",  # cow ambient sound
            "subtitles.entity.pig.ambient",  # pig ambient sound
            "subtitles.entity.chicken.ambient",  # chicken ambient sound
            "subtitles.entity.chicken.egg",  # chicken egg sound
        ]
