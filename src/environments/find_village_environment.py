import craftground
from environments.base_environment import BaseEnvironment
from wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


class FindVillageEnvironment(BaseEnvironment):
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
        class FindVillageWrapper(CleanUpFastResetWrapper):
            def __init__(self):
                self.env = craftground.make(
                    verbose=verbose,
                    env_path=env_path,
                    port=port,
                    initialInventoryCommands=["minecraft:firework_rocket 64"],
                    initialPosition=None,  # [0, , 0],  # nullable
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
                    miscStatKeys=["aviate_one_cm"],
                    isHudHidden=not hud,
                    initialExtraCommands=[
                        "item replace entity @p armor.chest with minecraft:elytra",
                        "difficulty peaceful",
                        "item replace entity @p armor.feet with minecraft:diamond_boots",
                    ],
                    surrounding_entities_keys=[30],
                    render_distance=render_distance,
                    simulation_distance=simulation_distance,
                )
                super(FindVillageWrapper, self).__init__(self.env)

        return FindVillageWrapper(), [
            "subtitles.entity.skeleton.ambient",  # TODO: remove
        ]
