import craftground
from environments.base_environment import BaseEnvironment


class FlatNightEnvironment(BaseEnvironment):
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
            render_distance=render_distance,
            simulation_distance=simulation_distance,
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
