import craftground
from environments.base_environment import BaseEnvironment


class FishingEnvironment(BaseEnvironment):
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
        return craftground.make(
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
            render_distance=render_distance,
            simulation_distance=simulation_distance,
        ), [
            "subtitles.entity.experience_orb.pickup",
            "subtitles.entity.fishing_bobber.retrieve",
            "subtitles.entity.fishing_bobber.splash",
            "subtitles.entity.fishing_bobber.throw",
            "subtitles.entity.item.pickup",
        ]
