import random

import gym
import numpy as np
from gymnasium.core import WrapperObsType

import mydojo


def make_husk_environment(verbose: bool, env_path: str, port: int):
    return mydojo.make(
        verbose=verbose,
        env_path=env_path,
        port=port,
        initialInventoryCommands=[],
        initialPosition=None,  # nullable
        initialMobsCommands=[
            # "minecraft:sheep",
            "minecraft:husk ~ ~ ~5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
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
    ), [
        "subtitles.entity.husk.ambient",
        "subtitles.block.generic.footsteps",
    ]


def make_husks_environment(verbose: bool, env_path: str, port: int):
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
    ), [
        "subtitles.entity.husk.ambient",
        "subtitles.block.generic.footsteps",
    ]


def make_husk_noisy_environment(verbose: bool, env_path: str, port: int):
    return mydojo.make(
        verbose=verbose,
        env_path=env_path,
        port=port,
        initialInventoryCommands=[],
        initialPosition=None,  # nullable
        initialMobsCommands=[
            "minecraft:sheep ~ ~ 5",
            "minecraft:cow ~ ~ -5",
            "minecraft:cow ~5 ~ -5",
            "minecraft:sheep ~-5 ~ -5",
            "minecraft:husk ~ ~ ~5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
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
        noisy=True,
    ), [
        "subtitles.entity.husk.ambient",
        "subtitles.entity.sheep.ambient",  # sheep ambient sound
        "subtitles.block.generic.footsteps",  # player, animal walking
        "subtitles.block.generic.break",  # sheep eating grass
        "subtitles.entity.cow.ambient",  # cow ambient sound
        # "subtitles.entity.pig.ambient",  # pig ambient sound
        # "subtitles.entity.chicken.ambient",  # chicken ambient sound
        # "subtitles.entity.chicken.egg",  # chicken egg sound
    ]


def make_husks_noisy_environment(verbose: bool, env_path: str, port: int):
    return mydojo.make(
        verbose=verbose,
        env_path=env_path,
        port=port,
        initialInventoryCommands=[],
        initialPosition=None,  # nullable
        initialMobsCommands=[
            "minecraft:sheep ~ ~ 5",
            "minecraft:cow ~ ~ -5",
            "minecraft:cow ~5 ~ -5",
            "minecraft:sheep ~-5 ~ -5",
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
        noisy=True,
    ), [
        "subtitles.entity.husk.ambient",
        "subtitles.entity.sheep.ambient",  # sheep ambient sound
        "subtitles.block.generic.footsteps",  # player, animal walking
        "subtitles.block.generic.break",  # sheep eating grass
        "subtitles.entity.cow.ambient",  # cow ambient sound
        # "subtitles.entity.pig.ambient",  # pig ambient sound
        # "subtitles.entity.chicken.ambient",  # chicken ambient sound
        # "subtitles.entity.chicken.egg",  # chicken egg sound
    ]


def make_husk_darkness_environment(verbose: bool, env_path: str, port: int):
    return mydojo.make(
        verbose=verbose,
        env_path=env_path,
        port=port,
        initialInventoryCommands=[],
        initialPosition=None,  # nullable
        initialMobsCommands=[
            "minecraft:husk ~ ~ ~5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
            # player looks at south (positive Z) when spawn
        ],
        imageSizeX=114,
        imageSizeY=64,
        visibleSizeX=114,
        visibleSizeY=64,
        seed=12345,  # nullable
        allowMobSpawn=False,
        alwaysDay=False,
        alwaysNight=True,
        initialWeather="clear",  # nullable
        isHardCore=False,
        isWorldFlat=True,  # superflat world
        obs_keys=["sound_subtitles"],
        initialExtraCommands=["effect give @p minecraft:darkness infinite 1 true"],
    ), [
        "subtitles.entity.husk.ambient",
        "subtitles.block.generic.footsteps",
    ]


def make_husks_darkness_environment(verbose: bool, env_path: str, port: int):
    return mydojo.make(
        verbose=verbose,
        env_path=env_path,
        port=port,
        initialInventoryCommands=[],
        initialPosition=None,  # nullable
        initialMobsCommands=[
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
        imageSizeX=114,
        imageSizeY=64,
        visibleSizeX=114,
        visibleSizeY=64,
        seed=12345,  # nullable
        allowMobSpawn=False,
        alwaysDay=False,
        alwaysNight=True,
        initialWeather="clear",  # nullable
        isHardCore=False,
        isWorldFlat=True,  # superflat world
        obs_keys=["sound_subtitles"],
        initialExtraCommands=["effect give @p minecraft:darkness infinite 1 true"],
    ), [
        "subtitles.entity.husk.ambient",
        "subtitles.block.generic.footsteps",
    ]


def make_find_animal_environment(verbose: bool, env_path: str, port: int):
    build_cage_comands = [
        "tp @p 0 -59 0",  # tp player
        "fill ~-15 ~-1 ~-15 ~15 ~2 ~15 minecraft:hay_block hollow",  # make a cage
        "fill ~-14 ~-1 ~-14 ~-11 ~-1 ~-11 minecraft:acacia_fence outline",  # make a cage
        "fill ~11 ~-1 ~11 ~14 ~-1 ~14 minecraft:acacia_fence outline",  # make a cage
        "fill ~-14 ~-1 ~11 ~-11 ~-1 ~14 minecraft:acacia_fence outline",  # make a cage
        "fill ~11 ~-1 ~-14 ~14 ~-1 ~-11 minecraft:acacia_fence outline",  # make a cage
        "fill ~-13 ~-1 ~-13 ~-12 ~-1 ~-12 minecraft:air outline",  # make a cage
        "fill ~12 ~-1 ~12 ~13 ~-1 ~13 minecraft:air outline",  # make a cage
        "fill ~-13 ~-1 ~12 ~-12 ~-1 ~13 minecraft:air outline",  # make a cage
        "fill ~12 ~-1 ~-13 ~13 ~-1 ~-12 minecraft:air outline",  # make a cage
        "fill ~-15 ~2 ~-15 ~15 ~10 ~15 minecraft:air replace",  # make a cage
    ]

    def summon_animal_commands(animal, x, z):
        return f"summon minecraft:{animal} ~{x} ~ ~{z}"

    coords = [
        (13, 13),
        (13, -13),
        (-13, 13),
        (-13, -13),
    ]
    random.shuffle(coords)
    summon_animal_commands_list = [
        summon_animal_commands("sheep", coords[0][0], coords[0][1]),
        summon_animal_commands("pig", coords[1][0], coords[1][1]),
        summon_animal_commands("chicken", coords[2][0], coords[2][1]),
    ] * 3

    return mydojo.make(
        verbose=verbose,
        env_path=env_path,
        port=port,
        initialInventoryCommands=[],
        initialPosition=None,  # nullable
        initialMobsCommands=[
            # "minecraft:sheep",
            # "minecraft:sheep ~ ~ ~4",
            # player looks at south (positive Z) when spawn
        ],
        imageSizeX=114,
        imageSizeY=64,
        visibleSizeX=342,
        visibleSizeY=192,
        seed=12345,  # nullable
        allowMobSpawn=False,
        alwaysDay=True,
        alwaysNight=False,
        initialWeather="clear",  # nullable
        isHardCore=False,
        isWorldFlat=True,  # superflat world
        obs_keys=["sound_subtitles"],
        initialExtraCommands=build_cage_comands + summon_animal_commands_list,
    ), [
        "subtitles.entity.sheep.ambient",  # sheep ambient sound
        "subtitles.block.generic.footsteps",  # player, animal walking
        "subtitles.block.generic.break",  # sheep eating grass
        "subtitles.entity.cow.ambient",  # cow ambient sound
        "subtitles.entity.pig.ambient",  # pig ambient sound
        "subtitles.entity.chicken.ambient",  # chicken ambient sound
        "subtitles.entity.chicken.egg",  # chicken egg sound
    ]


def make_random_husk_environment(verbose: bool, env_path: str, port: int):
    class RandomHuskWrapper(gym.Wrapper):
        def __init__(self):
            self.env = mydojo.make(
                verbose=verbose,
                env_path=env_path,
                port=port,
                initialInventoryCommands=[],
                initialPosition=None,  # nullable
                initialMobsCommands=[
                    "minecraft:husk ~ ~ ~5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
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
            super(RandomHuskWrapper, self).__init__(self.env)

        def reset(self, fast_reset: bool = True) -> WrapperObsType:
            dx = random.randint(-10, 10)
            dz = random.randint(-10, 10)
            obs = self.env.reset(
                fast_reset=fast_reset,
                extra_commands=[
                    "tp @e[type=!player] ~ -500 ~",
                    "summon minecraft:husk "
                    + f"~{dx} ~ ~{dz}"
                    + " {HandItems:[{Count:1,id:iron_shovel},{}]}",
                ],
            )
            print(f"dx={dx}, dz={dz}")
            return obs

    return RandomHuskWrapper(), [
        "subtitles.entity.husk.ambient",
        "subtitles.block.generic.footsteps",
    ]


env_makers = {
    "husk": make_husk_environment,
    "husks": make_husks_environment,
    "husk-noisy": make_husk_noisy_environment,
    "husks-noisy": make_husks_noisy_environment,
    "husk-darkness": make_husk_darkness_environment,
    "husks-darkness": make_husks_darkness_environment,
    "find-animal": make_find_animal_environment,
    "husk-random": make_random_husk_environment,
}
