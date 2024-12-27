# import pdb
from enum import Enum
from typing import List, Tuple, Optional

from .proto import initial_environment_pb2
from .screen_encoding_modes import ScreenEncodingMode


class GameMode(Enum):
    SURVIVAL = 0
    HARDCORE = 1
    CREATIVE = 2


class Difficulty(Enum):
    PEACEFUL = 0
    EASY = 1
    NORMAL = 2
    HARD = 3


class WorldType(Enum):
    DEFAULT = 0
    SUPERFLAT = 1
    LARGE_BIOMES = 2
    AMPLIFIED = 3
    SINGLE_BIOME = 4


class DaylightMode(Enum):
    DEFAULT = 0
    ALWAYS_DAY = 1
    ALWAYS_NIGHT = 2
    FREEZED = 3


class InitialEnvironmentConfig:
    def __init__(
        self,
        image_width=640,
        image_height=360,
        gamemode: GameMode = GameMode.SURVIVAL,
        difficulty: Difficulty = Difficulty.NORMAL,
        world_type: WorldType = WorldType.DEFAULT,
        world_type_args="",
        seed="",
        generate_structures=True,
        bonus_chest=False,
        datapack_paths=None,
        initial_extra_commands=None,
        killed_stat_keys=None,
        mined_stat_keys=None,
        misc_stat_keys=None,
        surrounding_entity_distances=None,
        hud_hidden: bool = False,
        render_distance: int = 6,
        simulation_distance: int = 6,
        eye_distance: float = 0.0,  # 0.1 is good for binocular
        structure_paths=None,
        no_fov_effect=False,
        request_raycast=False,
        screen_encoding_mode: ScreenEncodingMode = ScreenEncodingMode.RAW,
        requires_surrounding_blocks: bool = False,
        level_display_name_to_play: str = "",
        fov=70,
        requires_biome_info=False,
        requires_heightmap=False,
        **kwargs,
    ):
        self.imageSizeX = image_width
        self.imageSizeY = image_height
        self.gamemode = gamemode.value
        self.difficulty = difficulty.value
        self.world_type = world_type.value
        self.world_type_args = world_type_args
        # TODO: Parse world type args
        self.seed = seed
        self.generate_structures = generate_structures
        self.bonus_chest = bonus_chest
        self.datapack_paths = datapack_paths or []
        self.initialExtraCommands = initial_extra_commands or []
        self.killedStatKeys = killed_stat_keys or []
        self.minedStatKeys = mined_stat_keys or []
        self.miscStatKeys = misc_stat_keys or []
        self.surrounding_entities_keys = surrounding_entity_distances or []
        self.isHudHidden = hud_hidden
        self.render_distance = render_distance
        self.simulation_distance = simulation_distance
        self.eye_distance = eye_distance
        self.structure_paths = structure_paths or []
        self.no_fov_effect = no_fov_effect
        self.request_raycast = request_raycast
        self.screen_encoding_mode = screen_encoding_mode
        self.requires_surrounding_blocks = requires_surrounding_blocks
        self.level_display_name_to_play = level_display_name_to_play
        self.fov = fov
        self.requires_biome_info = requires_biome_info
        self.requires_heightmap = requires_heightmap
        if kwargs:
            print(f"Unexpected Kwargs: {kwargs}")

    def add_initial_mobs(self, mobs: List[str]):
        self.initialExtraCommands.extend([f"summon {mob}" for mob in mobs])
        return self

    def add_initial_inventory(self, items: List[Tuple[str, int]]):
        self.initialExtraCommands.extend(
            [f"give @p {item[0]} {item[1]}" for item in items]
        )
        return self

    # You need to specify as .0 if the precise value is required
    def set_initial_position(
        self,
        x: float,
        y: float,
        z: float,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
    ):
        if yaw is None and pitch is None:
            self.initialExtraCommands.append(f"tp @p {x} {y} {z}")
        else:
            assert (
                yaw is not None and pitch is not None
            ), "Both Yaw and Pitch must be set or not set at the same time"
            self.initialExtraCommands.append(f"tp @p {x} {y} {z} {yaw} {pitch}")
        return self

    def set_initial_weather(self, weather: str):
        self.initialExtraCommands.append(f"weather {weather}")
        return self

    def freeze_weather(self, freeze: bool = True):
        self.initialExtraCommands.append(
            f"gamerule doWeatherCycle {str(not freeze).lower()}"
        )
        return self

    def freeze_time(self, freeze: bool = True):
        self.initialExtraCommands.append(
            f"gamerule doDaylightCycle {str(not freeze).lower()}"
        )
        return self

    def set_allow_mob_spawn(self, allow: bool = True):
        self.initialExtraCommands.append(f"gamerule doMobSpawning {str(allow).lower()}")
        return self

    def set_daylight_cycle_mode(self, mode: DaylightMode):
        if mode == DaylightMode.ALWAYS_DAY:
            self.initialExtraCommands.append("time set day")
            self.initialExtraCommands.append("gamerule doDaylightCycle false")
        elif mode == DaylightMode.ALWAYS_NIGHT:
            self.initialExtraCommands.append("time set midnight")
            self.initialExtraCommands.append("gamerule doDaylightCycle false")
        elif mode == DaylightMode.FREEZED:
            self.initialExtraCommands.append("gamerule doDaylightCycle false")
        else:
            self.initialExtraCommands.append("gamerule doDaylightCycle true")
        return self

    def to_initial_environment_message(self):
        initial_env = initial_environment_pb2.InitialEnvironmentMessage()
        initial_env.imageSizeX = self.imageSizeX
        initial_env.imageSizeY = self.imageSizeY
        initial_env.gamemode = self.gamemode
        initial_env.difficulty = self.difficulty
        initial_env.worldType = self.world_type
        initial_env.worldTypeArgs = self.world_type_args
        initial_env.seed = self.seed
        initial_env.generate_structures = self.generate_structures
        initial_env.bonus_chest = self.bonus_chest
        if self.datapack_paths:
            initial_env.datapack_paths.extend(self.datapack_paths)
        if self.initialExtraCommands:
            initial_env.initialExtraCommands.extend(self.initialExtraCommands)
        if self.killedStatKeys:
            initial_env.killedStatKeys.extend(self.killedStatKeys)
        if self.minedStatKeys:
            initial_env.minedStatKeys.extend(self.minedStatKeys)
        if self.miscStatKeys:
            initial_env.miscStatKeys.extend(self.miscStatKeys)
        if self.surrounding_entities_keys:
            initial_env.surroundingEntityDistances.extend(
                self.surrounding_entities_keys
            )
        initial_env.hudHidden = self.isHudHidden
        initial_env.render_distance = self.render_distance
        initial_env.simulation_distance = self.simulation_distance
        initial_env.eye_distance = self.eye_distance
        if self.structure_paths:
            initial_env.structurePaths.extend(self.structure_paths)
        initial_env.no_fov_effect = self.no_fov_effect
        initial_env.request_raycast = self.request_raycast
        initial_env.screen_encoding_mode = self.screen_encoding_mode.value
        initial_env.requiresSurroundingBlocks = self.requires_surrounding_blocks
        initial_env.level_display_name_to_play = self.level_display_name_to_play
        initial_env.fov = self.fov
        initial_env.requiresBiomeInfo = self.requires_biome_info
        initial_env.requiresHeightmap = self.requires_heightmap
        return initial_env
