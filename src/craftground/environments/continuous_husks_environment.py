import random
from typing import Tuple, Optional, Any

from craftground import make
from gymnasium.core import ActType, ObsType
from gymnasium.core import WrapperObsType

from environments.base_environment import BaseEnvironment
from environments.husk_environment import generate_husks
from wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


# summons husks every 25 ticks
class ContinuousHuskEnvironment(BaseEnvironment):
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
        class RandomHuskWrapper(CleanUpFastResetWrapper):
            def __init__(self):
                initialExtraCommands = []
                initialExtraCommands.extend(generate_husks(1, 3, 5))
                self.env = make(
                    verbose=verbose,
                    env_path=env_path,
                    port=port,
                    initialInventoryCommands=[],
                    initialPosition=None,  # nullable
                    initialMobsCommands=[
                        # "minecraft:husk ~ ~ ~5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
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
                    initialExtraCommands=initialExtraCommands,
                    isHudHidden=not hud,
                    render_action=render_action,
                    render_distance=render_distance,
                    simulation_distance=simulation_distance,
                )
                super(RandomHuskWrapper, self).__init__(self.env)

            def reset(
                self,
                *,
                seed: Optional[int] = None,
                options: Optional[dict[str, Any]] = None,
            ) -> tuple[WrapperObsType, dict[str, Any]]:
                extra_commands = ["tp @e[type=!player] ~ -500 ~"]
                extra_commands.extend(generate_husks(1, 4, 7))

                obs = self.env.reset(
                    extra_commands=extra_commands,
                    seed=seed,
                    options=options,
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
