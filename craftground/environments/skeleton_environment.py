import random
from typing import Optional, Any, Tuple

from gymnasium.core import WrapperObsType

import craftground
from environments.base_environment import BaseEnvironment
from wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


class SkeletonEnvironment(BaseEnvironment):
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
        class RandomSkeletonWrapper(CleanUpFastResetWrapper):
            def __init__(self):
                self.env = craftground.make(
                    verbose=verbose,
                    env_path=env_path,
                    port=port,
                    initialInventoryCommands=[],
                    initialPosition=None,  # nullable
                    initialMobsCommands=[
                        "minecraft:skeleton ~ ~ ~5",
                        # player looks at south (positive Z) when spawn
                    ],
                    imageSizeX=size_x,
                    imageSizeY=size_y,
                    visibleSizeX=size_x,
                    visibleSizeY=size_y,
                    seed=12345,  # nullable
                    allowMobSpawn=False,
                    alwaysDay=False,
                    alwaysNight=True,
                    initialWeather="clear",  # nullable
                    isHardCore=False,
                    isWorldFlat=True,  # superflat world
                    obs_keys=["sound_subtitles"],
                    isHudHidden=not hud,
                    render_action=render_action,
                    render_distance=render_distance,
                    simulation_distance=simulation_distance,
                )
                super(RandomSkeletonWrapper, self).__init__(self.env)

            def reset(
                self,
                *,
                seed: Optional[int] = None,
                options: Optional[dict[str, Any]] = None,
            ) -> Tuple[WrapperObsType, dict[str, Any]]:
                dx = self.generate_random_excluding(-10, 10, -5, 5)
                dz = self.generate_random_excluding(-10, 10, -5, 5)
                options.update(
                    {
                        "fast_reset": True,
                        "extra_commands": [
                            "tp @e[type=!player] ~ -500 ~",
                            "summon minecraft:skeleton " + f"~{dx} ~ ~{dz}",
                        ],
                    }
                )
                obs, info = self.env.reset(seed=seed, options=options)
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
