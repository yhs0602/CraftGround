import random
from typing import Tuple, Optional, Dict, Any

from gymnasium.core import WrapperObsType, ActType, ObsType

import mydojo
from wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


def make_husk_environment(
    verbose: bool,
    env_path: str,
    port: int,
    size_x: int = 114,
    size_y: int = 64,
    hud: bool = False,
    render_action: bool = True,
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
            # player looks at south (positive Z) when spawn
        ],
        imageSizeX=size_x,
        imageSizeY=size_y,
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
        isHudHidden=not hud,
        render_action=render_action,
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


def make_husks_darkness_environment(
    verbose: bool, env_path: str, port: int, hud_hidden: bool = True
):
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
        isHudHidden=hud_hidden,
    ), [
        "subtitles.entity.husk.ambient",
        "subtitles.block.generic.footsteps",
    ]


def make_find_animal_environment(
    verbose: bool, env_path: str, port: int, hud_hidden: bool
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
            self.env = mydojo.make(
                verbose=verbose,
                env_path=env_path,
                port=port,
                initialInventoryCommands=[],
                initialPosition=None,  # nullable
                initialMobsCommands=[],
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
                initialExtraCommands=build_cage_comands + summon_animal_commands_list,
                surrounding_entities_keys=[1, 2, 5],
                isHudHidden=hud_hidden,
            )
            super(RandomAnimalWrapper, self).__init__(self.env)

        def reset(
            self,
            fast_reset: bool = True,
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
                fast_reset=fast_reset,
                extra_commands=extra_commands,
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


def make_random_husk_environment(
    verbose: bool,
    env_path: str,
    port: int,
    hud: bool = True,
    render_action: bool = True,
):
    class RandomHuskWrapper(CleanUpFastResetWrapper):
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
                isHudHidden=not hud,
                render_action=render_action,
            )
            super(RandomHuskWrapper, self).__init__(self.env)

        def reset(
            self, fast_reset: bool = True, **kwargs
        ) -> Tuple[ObsType, Dict[str, Any]]:
            dx = self.generate_random_excluding(-10, 10, -5, 5)
            dz = self.generate_random_excluding(-10, 10, -5, 5)
            obs, info = self.env.reset(
                fast_reset=fast_reset,
                extra_commands=[
                    "tp @e[type=!player] ~ -500 ~",
                    "summon minecraft:husk "
                    + f"~{dx} ~ ~{dz}"
                    + " {HandItems:[{Count:1,id:iron_shovel},{}]}",
                ],
                **kwargs,
            )
            print(f"dx={dx}, dz={dz}")
            obs["extra_info"] = {
                "husk_dx": dx,
                "husk_dz": dz,
            }
            return obs, info

        def generate_random_excluding(self, start, end, exclude_start, exclude_end):
            while True:
                x = random.randint(start, end)
                if x not in range(exclude_start, exclude_end):
                    return x

    return RandomHuskWrapper(), [
        "subtitles.entity.husk.ambient",
        "subtitles.block.generic.footsteps",
    ]


def make_random_husks_environment(
    verbose: bool, env_path: str, port: int, hud_hidden: bool
):
    class RandomHuskWrapper(CleanUpFastResetWrapper):
        def __init__(self):
            self.env = mydojo.make(
                verbose=verbose,
                env_path=env_path,
                port=port,
                initialInventoryCommands=[],
                initialPosition=None,  # nullable
                initialMobsCommands=[
                    # "minecraft:husk ~ ~ ~5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
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
                initialExtraCommands=generate_husks(5, 5, 10),
                isHudHidden=hud_hidden,
            )
            super(RandomHuskWrapper, self).__init__(self.env)

        def reset(self, fast_reset: bool = True, **kwargs) -> WrapperObsType:
            extra_commands = ["tp @e[type=!player] ~ -500 ~"]

            gen_husk_commands = generate_husks(5, 5, 10)
            extra_commands.extend(gen_husk_commands)

            obs = self.env.reset(
                fast_reset=fast_reset,
                extra_commands=extra_commands,
                **kwargs,
            )
            # obs["extra_info"] = {
            #     "husk_dx": dx,
            #     "husk_dz": dz,
            # }
            return obs

    return RandomHuskWrapper(), [
        "subtitles.entity.husk.ambient",
        "subtitles.block.generic.footsteps",
    ]


def make_random_husks_darkness_environment(
    verbose: bool, env_path: str, port: int, hud_hidden: bool = True
):
    class RandomHuskWrapper(CleanUpFastResetWrapper):
        def __init__(self):
            initialExtraCommands = ["effect give @p minecraft:darkness infinite 1 true"]
            initialExtraCommands.extend(generate_husks(40, 5, 10))
            self.env = mydojo.make(
                verbose=verbose,
                env_path=env_path,
                port=port,
                initialInventoryCommands=[],
                initialPosition=None,  # nullable
                initialMobsCommands=[
                    # "minecraft:husk ~ ~ ~5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
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
                initialExtraCommands=initialExtraCommands,
                isHudHidden=hud_hidden,
            )
            super(RandomHuskWrapper, self).__init__(self.env)

        def reset(self, fast_reset: bool = True, **kwargs) -> WrapperObsType:
            extra_commands = ["tp @e[type=!player] ~ -500 ~"]
            extra_commands.extend(generate_husks(10, 5, 10))

            obs = self.env.reset(
                fast_reset=fast_reset,
                extra_commands=extra_commands,
            )
            # obs["extra_info"] = {
            #     "husk_dx": dx,
            #     "husk_dz": dz,
            # }
            return obs

    return RandomHuskWrapper(), [
        "subtitles.entity.husk.ambient",
        "subtitles.block.generic.footsteps",
    ]


# summons husks every 25 ticks
def make_continuous_husks_environment(
    verbose: bool, env_path: str, port: int, hud_hidden: bool = True
):
    class RandomHuskWrapper(CleanUpFastResetWrapper):
        def __init__(self):
            initialExtraCommands = []
            initialExtraCommands.extend(generate_husks(1, 3, 5))
            self.env = mydojo.make(
                verbose=verbose,
                env_path=env_path,
                port=port,
                initialInventoryCommands=[],
                initialPosition=None,  # nullable
                initialMobsCommands=[
                    # "minecraft:husk ~ ~ ~5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
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
                initialExtraCommands=initialExtraCommands,
                isHudHidden=hud_hidden,
            )
            super(RandomHuskWrapper, self).__init__(self.env)

        def reset(self, fast_reset: bool = True, **kwargs) -> WrapperObsType:
            extra_commands = ["tp @e[type=!player] ~ -500 ~"]
            extra_commands.extend(generate_husks(1, 4, 7))

            obs = self.env.reset(
                fast_reset=fast_reset, extra_commands=extra_commands, **kwargs
            )
            # obs["extra_info"] = {
            #     "husk_dx": dx,
            #     "husk_dz": dz,
            # }
            return obs

        def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
            obs, reward, terminated, truncated, info = self.env.step(action)
            if random.randint(0, 50) == 0:
                extra_commands = generate_husks(1, 4, 7)
                self.env.add_commands(extra_commands)
            return obs, reward, terminated, truncated, info

    return RandomHuskWrapper(), [
        "subtitles.entity.husk.ambient",
        "subtitles.block.generic.footsteps",
    ]


def make_random_husk_terrain_environment(
    verbose: bool,
    env_path: str,
    port: int,
    darkness: bool = False,
    hud_hidden: bool = True,
):
    class RandomHuskWrapper(CleanUpFastResetWrapper):
        def __init__(self):
            initialExtraCommands = []
            initialExtraCommands.extend(generate_husks(1, 5, 10, dy=8))
            if darkness:
                initialExtraCommands.append(
                    "effect give @p minecraft:darkness infinite 1 true"
                )
            self.env = mydojo.make(
                verbose=verbose,
                env_path=env_path,
                port=port,
                initialInventoryCommands=[],
                initialPosition=None,  # nullable
                initialMobsCommands=[
                    # "minecraft:husk ~ ~ ~5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
                    # player looks at south (positive Z) when spawn
                ],
                imageSizeX=114,
                imageSizeY=64,
                visibleSizeX=114,
                visibleSizeY=64,
                seed=3788863154090864390,  # nullable
                allowMobSpawn=False,
                alwaysDay=True,
                alwaysNight=False,
                initialWeather="clear",  # nullable
                isHardCore=False,
                isWorldFlat=False,  # superflat world
                obs_keys=["sound_subtitles"],
                initialExtraCommands=initialExtraCommands,
                isHudHidden=hud_hidden,
            )
            super(RandomHuskWrapper, self).__init__(self.env)

        def reset(
            self,
            fast_reset: bool = True,
            seed: Optional[int] = None,
            options: Optional[dict[str, Any]] = None,
        ) -> tuple[WrapperObsType, dict[str, Any]]:
            extra_commands = ["tp @e[type=!player] ~ -500 ~"]
            extra_commands.extend(generate_husks(1, 5, 10, dy=8))

            obs = self.env.reset(
                fast_reset=fast_reset,
                extra_commands=extra_commands,
                seed=seed,
                options=options,
            )
            # obs["extra_info"] = {
            #     "husk_dx": dx,
            #     "husk_dz": dz,
            # }
            return obs

    return RandomHuskWrapper(), [
        "subtitles.entity.husk.ambient",
        "subtitles.block.generic.footsteps",
    ]


def make_random_husk_forest_environment(
    verbose: bool,
    env_path: str,
    port: int,
    darkness: bool = False,
    hud_hidden: bool = True,
):
    class RandomHuskWrapper(CleanUpFastResetWrapper):
        def __init__(self):
            initialExtraCommands = []
            initialExtraCommands.extend(generate_husks(1, 5, 10, dy=8))
            if darkness:
                initialExtraCommands.append(
                    "effect give @p minecraft:darkness infinite 1 true"
                )
            self.env = mydojo.make(
                verbose=verbose,
                env_path=env_path,
                port=port,
                initialInventoryCommands=[],
                initialPosition=[-117, 75, -15],  # nullable
                initialMobsCommands=[
                    # "minecraft:husk ~ ~ ~5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
                    # player looks at south (positive Z) when spawn
                ],
                imageSizeX=114,
                imageSizeY=64,
                visibleSizeX=114,
                visibleSizeY=64,
                seed=3788863154090864390,  # nullable
                allowMobSpawn=False,
                alwaysDay=True,
                alwaysNight=False,
                initialWeather="clear",  # nullable
                isHardCore=False,
                isWorldFlat=False,  # superflat world
                obs_keys=["sound_subtitles"],
                initialExtraCommands=initialExtraCommands,
                isHudHidden=hud_hidden,
            )
            super(RandomHuskWrapper, self).__init__(self.env)

        def reset(
            self,
            fast_reset: bool = True,
            seed: Optional[int] = None,
            options: Optional[dict[str, Any]] = None,
        ) -> tuple[WrapperObsType, dict[str, Any]]:
            extra_commands = ["tp @e[type=!player] ~ -500 ~"]
            extra_commands.extend(generate_husks(1, 5, 10, dy=8))

            obs = self.env.reset(
                fast_reset=fast_reset,
                extra_commands=extra_commands,
                seed=seed,
                options=options,
            )
            # obs["extra_info"] = {
            #     "husk_dx": dx,
            #     "husk_dz": dz,
            # }
            return obs

    return RandomHuskWrapper(), [
        "subtitles.entity.husk.ambient",
        "subtitles.block.generic.footsteps",
    ]


def make_hunt_husk_environment(
    verbose: bool,
    env_path: str,
    port: int,
    hud: bool = False,
    render_action: bool = False,
):
    class RandomHuskWrapper(CleanUpFastResetWrapper):
        def __init__(self):
            initialExtraCommands = []
            initialExtraCommands.extend(generate_husks(1, 5, 10))
            initialExtraCommands.extend(
                ["item replace entity @p weapon.offhand with minecraft:shield"]
            )
            self.env = mydojo.make(
                verbose=verbose,
                env_path=env_path,
                port=port,
                render_action=render_action,
                initialInventoryCommands=[
                    "minecraft:diamond_sword",
                ],
                initialPosition=None,  # nullable
                initialMobsCommands=[],
                imageSizeX=114,
                imageSizeY=64,
                visibleSizeX=114,
                visibleSizeY=64,
                seed=3788863154090864390,  # nullable
                allowMobSpawn=False,
                alwaysDay=True,
                alwaysNight=False,
                initialWeather="clear",  # nullable
                isHardCore=False,
                isWorldFlat=True,  # superflat world
                obs_keys=["sound_subtitles"],
                initialExtraCommands=initialExtraCommands,
                killedStatKeys=["minecraft:husk"],
                isHudHidden=not hud,
            )
            super(RandomHuskWrapper, self).__init__(self.env)

        def reset(
            self,
            fast_reset: bool = True,
            seed: Optional[int] = None,
            options: Optional[dict[str, Any]] = None,
        ) -> tuple[WrapperObsType, dict[str, Any]]:
            extra_commands = ["tp @e[type=!player] ~ -500 ~"]
            extra_commands.extend(generate_husks(1, 5, 10))

            obs = self.env.reset(
                fast_reset=fast_reset,
                extra_commands=extra_commands,
                seed=seed,
                options=options,
            )
            # obs["extra_info"] = {
            #     "husk_dx": dx,
            #     "husk_dz": dz,
            # }
            return obs

    return RandomHuskWrapper(), [
        "subtitles.entity.husk.ambient",
        "subtitles.block.generic.footsteps",
        "subtitles.entity.player.attack.crit",
        "subtitles.entity.player.attack.knockback",
        "subtitles.entity.player.attack.strong",
        "subtitles.entity.player.attack.sweep",
        "subtitles.entity.player.attack.weak",
        "subtitles.entity.husk.hurt",
        "subtitles.item.shield.block",
    ]


def make_mansion_environment(verbose: bool, env_path: str, port: int, hud_hidden: bool):
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

    return mydojo.make(
        verbose=verbose,
        env_path=env_path,
        port=port,
        initialInventoryCommands=[],
        initialPosition=[-32, 78, -35],  # nullable
        initialMobsCommands=[],
        imageSizeX=114,
        imageSizeY=64,
        visibleSizeX=114,
        visibleSizeY=64,
        seed=8952232712572833477,  # nullable
        allowMobSpawn=False,
        alwaysDay=True,
        alwaysNight=False,
        initialWeather="clear",  # nullable
        isHardCore=False,
        isWorldFlat=False,  # superflat world
        obs_keys=["sound_subtitles"],
        initialExtraCommands=build_mansion,
        isHudHidden=hud_hidden,
    ), [
        "subtitles.block.campfire.crackle",  # Campfire crackles
    ]


def make_skeleton_random_environment(
    verbose: bool, env_path: str, port: int, hud_hidden: bool = True
):
    class RandomSkeletonWrapper(CleanUpFastResetWrapper):
        def __init__(self):
            self.env = mydojo.make(
                verbose=verbose,
                env_path=env_path,
                port=port,
                initialInventoryCommands=[],
                initialPosition=None,  # nullable
                initialMobsCommands=[
                    "minecraft:skeleton ~ ~ ~5",
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
                isHudHidden=hud_hidden,
            )
            super(RandomSkeletonWrapper, self).__init__(self.env)

        def reset(
            self, fast_reset: bool = True, **kwargs
        ) -> Tuple[ObsType, Dict[str, Any]]:
            dx = self.generate_random_excluding(-10, 10, -5, 5)
            dz = self.generate_random_excluding(-10, 10, -5, 5)
            obs, info = self.env.reset(
                fast_reset=fast_reset,
                extra_commands=[
                    "tp @e[type=!player] ~ -500 ~",
                    "summon minecraft:skeleton " + f"~{dx} ~ ~{dz}",
                ],
                **kwargs,
            )
            print(f"dx={dx}, dz={dz}")
            obs["extra_info"] = {
                "skeleton_dx": dx,
                "skeleton_dz": dz,
            }
            return obs, info

        def generate_random_excluding(self, start, end, exclude_start, exclude_end):
            while True:
                x = random.randint(start, end)
                if x not in range(exclude_start, exclude_end):
                    return x

    return RandomSkeletonWrapper(), [
        "subtitles.entity.skeleton.ambient",
        "subtitles.entity.skeleton.shoot",
        "subtitles.entity.arrow.hit_player",
        "subtitles.entity.arrow.hit",
        "subtitles.block.generic.footsteps",
    ]


def make_find_village_environment(
    verbose: bool, env_path: str, port: int, hud_hidden: bool = True
):
    class FindVillageWrapper(CleanUpFastResetWrapper):
        def __init__(self):
            self.env = mydojo.make(
                verbose=verbose,
                env_path=env_path,
                port=port,
                initialInventoryCommands=["minecraft:firework_rocket 64"],
                initialPosition=None,  # [0, , 0],  # nullable
                initialMobsCommands=[],
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
                miscStatKeys=["aviate_one_cm"],
                isHudHidden=hud_hidden,
                initialExtraCommands=[
                    "item replace entity @p armor.chest with minecraft:elytra",
                    "difficulty peaceful",
                    "item replace entity @p armor.feet with minecraft:diamond_boots",
                ],
                surrounding_entities_keys=[30],
            )
            super(FindVillageWrapper, self).__init__(self.env)

    return FindVillageWrapper(), [
        "subtitles.entity.skeleton.ambient",  # TODO: remove
    ]


def make_flat_night_environment(
    verbose: bool,
    env_path: str,
    port: int,
    size_x: int = 114,
    size_y: int = 64,
    hud: bool = False,
    render_action: bool = True,
):
    return mydojo.make(
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
        allowMobSpawn=True,
        alwaysDay=False,
        alwaysNight=True,
        initialWeather="clear",  # nullable
        isHardCore=False,
        isWorldFlat=True,  # superflat world
        obs_keys=["sound_subtitles"],
        initialExtraCommands=["difficulty normal", "gamerule mobGriefing false"],
        isHudHidden=not hud,
        render_action=render_action,
    ), [
        "subtitles.entity.slime.attack",
        "subtitles.entity.slime.death",
        "subtitles.entity.slime.hurt",
        "subtitles.entity.slime.squish",
        "entity.skeleton.ambient",
        "entity.skeleton.death",
        "entity.skeleton.hurt",
        "entity.skeleton.shoot",
        "entity.skeleton.step",
        "subtitles.entity.arrow.hit_player",
        "subtitles.entity.arrow.hit",
        "subtitles.entity.zombie.ambient",
        "subtitles.entity.zombie.death",
        "subtitles.entity.zombie.hurt",
        "subtitles.entity.creeper.death",
        "subtitles.entity.creeper.hurt",
        "subtitles.entity.creeper.primed",
        "subtitles.entity.generic.explode",
        "subtitles.entity.spider.ambient",
        "subtitles.entity.spider.death",
        "subtitles.entity.spider.hurt",
        "subtitles.entity.witch.ambient",
        "subtitles.entity.witch.drink",
        "subtitles.entity.witch.death",
        "subtitles.entity.witch.hurt",
        "subtitles.entity.witch.throw",
        "subtitles.entity.potion.splash",
        "subtitles.block.generic.footsteps",
    ]


def make_fishing_environment(
    verbose: bool,
    env_path: str,
    port: int,
    size_x: int = 114,
    size_y: int = 64,
    hud: bool = False,
    render_action: bool = True,
):
    return mydojo.make(
        verbose=verbose,
        env_path=env_path,
        port=port,
        initialInventoryCommands=[
            "fishing_rod{Enchantments:[{id:lure,lvl:3},{id:mending,lvl:1},{id:unbreaking,lvl:3}]} 1"
        ],
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
        initialWeather="rain",  # nullable
        isHardCore=False,
        isWorldFlat=False,  # superflat world
        obs_keys=["sound_subtitles"],
        miscStatKeys=["fish_caught"],
        initialExtraCommands=["tp @p -25 62 -277 127.2 -6.8"],  # x y z yaw pitch
        isHudHidden=not hud,
        render_action=render_action,
    ), [
        "subtitles.entity.experience_orb.pickup",
        "subtitles.entity.fishing_bobber.retrieve",
        "subtitles.entity.fishing_bobber.splash",
        "subtitles.entity.fishing_bobber.throw",
        "subtitles.entity.item.pickup",
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
    "husks-random": make_random_husks_environment,
    "husks-random-darkness": make_random_husks_darkness_environment,
    "husks-continuous": make_continuous_husks_environment,
    "husk-random-terrain": make_random_husk_terrain_environment,
    "husk-random-forest": make_random_husk_forest_environment,
    "husk-hunt": make_hunt_husk_environment,
    "mansion": make_mansion_environment,
    "skeleton-random": make_skeleton_random_environment,
    "find-village": make_find_village_environment,
    "flat-night": make_flat_night_environment,
    "fishing": make_fishing_environment,
}


def generate_husks(
    num_husks,
    min_distnace,
    max_distance,
    dy: Optional[int] = None,
    is_baby: bool = False,
    shovel: bool = False,
):
    commands = []
    success_count = 0
    is_baby_int = 1 if is_baby else 0
    while success_count < num_husks:
        dx = generate_random(-max_distance, max_distance)
        dz = generate_random(-max_distance, max_distance)
        if dy is None:
            dy = 0
        if dx * dx + dz * dz + dy * dy < min_distnace * min_distnace:
            continue
        shovel_command = "HandItems:[{Count:1,id:iron_shovel},{}],"
        health_command = 'Health:5,Attributes:[{Name:"generic.max_health",Base:5f}],'
        if not shovel:
            shovel_command = ""
        commands.append(
            "summon minecraft:husk "
            + f"~{dx} ~{dy} ~{dz}"
            + " {"
            # + health_command
            + shovel_command
            + f" IsBaby:{is_baby_int}"
            + "}"
        )
        success_count += 1
        print(f"dx={dx}, dz={dz}")
    return commands


def generate_random(start, end):
    return random.randint(start, end)


def generate_random_excluding(start, end, exclude_start, exclude_end):
    while True:
        x = random.randint(start, end)
        if x not in range(exclude_start, exclude_end):
            return x
