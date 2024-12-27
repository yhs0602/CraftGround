import random
from typing import Optional, Any, Tuple

from gymnasium.core import WrapperObsType

import craftground
from environments.base_environment import BaseEnvironment
from wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


class HuskEnvironment(BaseEnvironment):
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
        num_husks: int = 1,
        random_pos: bool = True,  # randomize husk position
        darkness: bool = False,  # add darkness effect
        strong: bool = False,  # give husk strong shovel
        noisy: bool = False,  # add noisy mobs
        is_baby: bool = False,  # make husk baby
        terrain: int = 0,  # 0: flat, 1: random, 2: random with water
        can_hunt: bool = False,  # player can hunt husks
        *args,
        **kwargs,
    ):
        if darkness:
            darkness_commands = ["effect give @p minecraft:darkness infinite 1 true"]
        else:
            darkness_commands = []
        if noisy:
            mobs_commands = [
                "minecraft:sheep ~ ~ 5",
                "minecraft:cow ~ ~ -5",
                "minecraft:cow ~5 ~ -5",
                "minecraft:sheep ~-5 ~ -5",
            ]
            noisy_sounds = [
                "subtitles.entity.sheep.ambient",  # sheep ambient sound
                "subtitles.block.generic.footsteps",  # player, animal walking
                "subtitles.block.generic.break",  # sheep eating grass
                "subtitles.entity.cow.ambient",  # cow ambient sound
            ]
        else:
            mobs_commands = []
            noisy_sounds = []
        husks_commands = generate_husks(
            num_husks,
            5,
            10,
            shovel=strong,
            is_baby=is_baby,
            randomize=random_pos,
        )
        killedStatKeys = []
        if can_hunt:
            inventory_commands = [
                "item replace entity @p weapon.offhand with minecraft:shield",
                "give @p minecraft:diamond_sword",
            ]
            killedStatKeys = (["minecraft:husk"],)
            hunt_sounds = [
                "subtitles.entity.player.attack.crit",
                "subtitles.entity.player.attack.knockback",
                "subtitles.entity.player.attack.strong",
                "subtitles.entity.player.attack.sweep",
                "subtitles.entity.player.attack.weak",
                "subtitles.entity.husk.hurt",
                "subtitles.item.shield.block",
            ]

        initial_extra_commands = (
            darkness_commands + mobs_commands + husks_commands + inventory_commands
        )
        sounds = (
            [
                "subtitles.entity.husk.ambient",
                "subtitles.block.generic.footsteps",
            ]
            + noisy_sounds
            + hunt_sounds
        )

        if terrain == 0:  # flat
            seed = 12345
            initialPosition = None
            isWorldFlat = True
        elif terrain == 1:  # terrain
            seed = 3788863154090864390
            initialPosition = None
            isWorldFlat = False
        elif terrain == 2:  # forest
            seed = 3788863154090864390
            initialPosition = [-117, 75, -15]
            isWorldFlat = False

        class RandomHuskWrapper(CleanUpFastResetWrapper):
            def __init__(self):
                self.env = craftground.make(
                    verbose=verbose,
                    env_path=env_path,
                    port=port,
                    initial_env_config=craftground.InitialEnvironmentConfig(
                        initialInventoryCommands=[],
                        initialPosition=initialPosition,  # nullable
                        initialMobsCommands=[
                            # "minecraft:husk ~ ~ ~5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
                            # player looks at south (positive Z) when spawn
                        ],
                        imageSizeX=size_x,
                        imageSizeY=size_y,
                        visibleSizeX=size_x,
                        visibleSizeY=size_y,
                        seed=seed,  # nullable
                        allowMobSpawn=False,
                        alwaysDay=True,
                        alwaysNight=False,
                        initialWeather="clear",  # nullable
                        isHardCore=False,
                        isWorldFlat=isWorldFlat,  # superflat world
                        obs_keys=["sound_subtitles"],
                        initialExtraCommands=initial_extra_commands,
                        isHudHidden=not hud,
                        render_action=render_action,
                        render_distance=render_distance,
                        simulation_distance=simulation_distance,
                        killed_stat_keys=killedStatKeys,
                    ),
                )
                super(RandomHuskWrapper, self).__init__(self.env)

            def reset(
                self,
                *,
                seed: Optional[int] = None,
                options: Optional[dict[str, Any]] = None,
            ) -> Tuple[WrapperObsType, dict[str, Any]]:
                extra_commands = ["tp @e[type=!player] ~ -500 ~"]
                extra_commands.extend(initial_extra_commands)
                options.update(
                    {
                        "extra_commands": extra_commands,
                    }
                )
                obs = self.env.reset(
                    seed=seed,
                    options=options,
                )
                # obs["extra_info"] = {
                #     "husk_dx": dx,
                #     "husk_dz": dz,
                # }
                return obs

        return RandomHuskWrapper(), sounds


def generate_husks(
    num_husks,
    min_distnace,
    max_distance,
    dy: Optional[int] = None,
    is_baby: bool = False,
    shovel: bool = False,
    randomize: bool = True,
    reduce_zombie_health: bool = False,
):
    commands = []
    success_count = 0
    is_baby_int = 1 if is_baby else 0
    while success_count < num_husks:
        if randomize:
            dx = generate_random(-max_distance, max_distance)
            dz = generate_random(-max_distance, max_distance)
        else:
            dx = 0
            dz = 5
        if dy is None:
            dy = 0
        if dx * dx + dz * dz + dy * dy < min_distnace * min_distnace:
            continue
        shovel_command = "HandItems:[{Count:1,id:iron_shovel},{}],"
        health_command = (
            'Health:5,Attributes:[{Name:"generic.max_health",Base:5f}],'
            if reduce_zombie_health
            else ""
        )
        if not shovel:
            shovel_command = ""
        commands.append(
            "summon minecraft:husk "
            + f"~{dx} ~{dy} ~{dz}"
            + " {"
            + health_command
            + shovel_command
            + f" IsBaby:{is_baby_int}"
            + "}"
        )
        success_count += 1
        print(f"dx={dx}, dz={dz}")
    return commands


def generate_random(start, end):
    return random.randint(start, end)
