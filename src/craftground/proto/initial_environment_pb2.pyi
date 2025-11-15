from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GameMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SURVIVAL: _ClassVar[GameMode]
    HARDCORE: _ClassVar[GameMode]
    CREATIVE: _ClassVar[GameMode]

class Difficulty(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PEACEFUL: _ClassVar[Difficulty]
    EASY: _ClassVar[Difficulty]
    NORMAL: _ClassVar[Difficulty]
    HARD: _ClassVar[Difficulty]

class WorldType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    DEFAULT: _ClassVar[WorldType]
    SUPERFLAT: _ClassVar[WorldType]
    LARGE_BIOMES: _ClassVar[WorldType]
    AMPLIFIED: _ClassVar[WorldType]
    SINGLE_BIOME: _ClassVar[WorldType]
SURVIVAL: GameMode
HARDCORE: GameMode
CREATIVE: GameMode
PEACEFUL: Difficulty
EASY: Difficulty
NORMAL: Difficulty
HARD: Difficulty
DEFAULT: WorldType
SUPERFLAT: WorldType
LARGE_BIOMES: WorldType
AMPLIFIED: WorldType
SINGLE_BIOME: WorldType

class LidarConfig(_message.Message):
    __slots__ = ("horizontal_rays", "max_distance", "vertical_angle", "vertical_rays", "vertical_fov")
    HORIZONTAL_RAYS_FIELD_NUMBER: _ClassVar[int]
    MAX_DISTANCE_FIELD_NUMBER: _ClassVar[int]
    VERTICAL_ANGLE_FIELD_NUMBER: _ClassVar[int]
    VERTICAL_RAYS_FIELD_NUMBER: _ClassVar[int]
    VERTICAL_FOV_FIELD_NUMBER: _ClassVar[int]
    horizontal_rays: int
    max_distance: float
    vertical_angle: float
    vertical_rays: int
    vertical_fov: float
    def __init__(self, horizontal_rays: _Optional[int] = ..., max_distance: _Optional[float] = ..., vertical_angle: _Optional[float] = ..., vertical_rays: _Optional[int] = ..., vertical_fov: _Optional[float] = ...) -> None: ...

class InitialEnvironmentMessage(_message.Message):
    __slots__ = ("imageSizeX", "imageSizeY", "gamemode", "difficulty", "worldType", "worldTypeArgs", "seed", "generate_structures", "bonus_chest", "datapackPaths", "initialExtraCommands", "killedStatKeys", "minedStatKeys", "miscStatKeys", "surroundingEntityDistances", "hudHidden", "render_distance", "simulation_distance", "eye_distance", "structurePaths", "no_fov_effect", "request_raycast", "screen_encoding_mode", "requiresSurroundingBlocks", "level_display_name_to_play", "fov", "requiresBiomeInfo", "requiresHeightmap", "python_pid", "requiresDepth", "requiresDepthConversion", "resourceZipPath", "blockCollisionKeys", "entityCollisionKeys", "map_dir_path", "lidar_config")
    IMAGESIZEX_FIELD_NUMBER: _ClassVar[int]
    IMAGESIZEY_FIELD_NUMBER: _ClassVar[int]
    GAMEMODE_FIELD_NUMBER: _ClassVar[int]
    DIFFICULTY_FIELD_NUMBER: _ClassVar[int]
    WORLDTYPE_FIELD_NUMBER: _ClassVar[int]
    WORLDTYPEARGS_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    GENERATE_STRUCTURES_FIELD_NUMBER: _ClassVar[int]
    BONUS_CHEST_FIELD_NUMBER: _ClassVar[int]
    DATAPACKPATHS_FIELD_NUMBER: _ClassVar[int]
    INITIALEXTRACOMMANDS_FIELD_NUMBER: _ClassVar[int]
    KILLEDSTATKEYS_FIELD_NUMBER: _ClassVar[int]
    MINEDSTATKEYS_FIELD_NUMBER: _ClassVar[int]
    MISCSTATKEYS_FIELD_NUMBER: _ClassVar[int]
    SURROUNDINGENTITYDISTANCES_FIELD_NUMBER: _ClassVar[int]
    HUDHIDDEN_FIELD_NUMBER: _ClassVar[int]
    RENDER_DISTANCE_FIELD_NUMBER: _ClassVar[int]
    SIMULATION_DISTANCE_FIELD_NUMBER: _ClassVar[int]
    EYE_DISTANCE_FIELD_NUMBER: _ClassVar[int]
    STRUCTUREPATHS_FIELD_NUMBER: _ClassVar[int]
    NO_FOV_EFFECT_FIELD_NUMBER: _ClassVar[int]
    REQUEST_RAYCAST_FIELD_NUMBER: _ClassVar[int]
    SCREEN_ENCODING_MODE_FIELD_NUMBER: _ClassVar[int]
    REQUIRESSURROUNDINGBLOCKS_FIELD_NUMBER: _ClassVar[int]
    LEVEL_DISPLAY_NAME_TO_PLAY_FIELD_NUMBER: _ClassVar[int]
    FOV_FIELD_NUMBER: _ClassVar[int]
    REQUIRESBIOMEINFO_FIELD_NUMBER: _ClassVar[int]
    REQUIRESHEIGHTMAP_FIELD_NUMBER: _ClassVar[int]
    PYTHON_PID_FIELD_NUMBER: _ClassVar[int]
    REQUIRESDEPTH_FIELD_NUMBER: _ClassVar[int]
    REQUIRESDEPTHCONVERSION_FIELD_NUMBER: _ClassVar[int]
    RESOURCEZIPPATH_FIELD_NUMBER: _ClassVar[int]
    BLOCKCOLLISIONKEYS_FIELD_NUMBER: _ClassVar[int]
    ENTITYCOLLISIONKEYS_FIELD_NUMBER: _ClassVar[int]
    MAP_DIR_PATH_FIELD_NUMBER: _ClassVar[int]
    LIDAR_CONFIG_FIELD_NUMBER: _ClassVar[int]
    imageSizeX: int
    imageSizeY: int
    gamemode: GameMode
    difficulty: Difficulty
    worldType: WorldType
    worldTypeArgs: str
    seed: str
    generate_structures: bool
    bonus_chest: bool
    datapackPaths: _containers.RepeatedScalarFieldContainer[str]
    initialExtraCommands: _containers.RepeatedScalarFieldContainer[str]
    killedStatKeys: _containers.RepeatedScalarFieldContainer[str]
    minedStatKeys: _containers.RepeatedScalarFieldContainer[str]
    miscStatKeys: _containers.RepeatedScalarFieldContainer[str]
    surroundingEntityDistances: _containers.RepeatedScalarFieldContainer[int]
    hudHidden: bool
    render_distance: int
    simulation_distance: int
    eye_distance: float
    structurePaths: _containers.RepeatedScalarFieldContainer[str]
    no_fov_effect: bool
    request_raycast: bool
    screen_encoding_mode: int
    requiresSurroundingBlocks: bool
    level_display_name_to_play: str
    fov: float
    requiresBiomeInfo: bool
    requiresHeightmap: bool
    python_pid: int
    requiresDepth: bool
    requiresDepthConversion: bool
    resourceZipPath: str
    blockCollisionKeys: _containers.RepeatedScalarFieldContainer[str]
    entityCollisionKeys: _containers.RepeatedScalarFieldContainer[str]
    map_dir_path: str
    lidar_config: LidarConfig
    def __init__(self, imageSizeX: _Optional[int] = ..., imageSizeY: _Optional[int] = ..., gamemode: _Optional[_Union[GameMode, str]] = ..., difficulty: _Optional[_Union[Difficulty, str]] = ..., worldType: _Optional[_Union[WorldType, str]] = ..., worldTypeArgs: _Optional[str] = ..., seed: _Optional[str] = ..., generate_structures: bool = ..., bonus_chest: bool = ..., datapackPaths: _Optional[_Iterable[str]] = ..., initialExtraCommands: _Optional[_Iterable[str]] = ..., killedStatKeys: _Optional[_Iterable[str]] = ..., minedStatKeys: _Optional[_Iterable[str]] = ..., miscStatKeys: _Optional[_Iterable[str]] = ..., surroundingEntityDistances: _Optional[_Iterable[int]] = ..., hudHidden: bool = ..., render_distance: _Optional[int] = ..., simulation_distance: _Optional[int] = ..., eye_distance: _Optional[float] = ..., structurePaths: _Optional[_Iterable[str]] = ..., no_fov_effect: bool = ..., request_raycast: bool = ..., screen_encoding_mode: _Optional[int] = ..., requiresSurroundingBlocks: bool = ..., level_display_name_to_play: _Optional[str] = ..., fov: _Optional[float] = ..., requiresBiomeInfo: bool = ..., requiresHeightmap: bool = ..., python_pid: _Optional[int] = ..., requiresDepth: bool = ..., requiresDepthConversion: bool = ..., resourceZipPath: _Optional[str] = ..., blockCollisionKeys: _Optional[_Iterable[str]] = ..., entityCollisionKeys: _Optional[_Iterable[str]] = ..., map_dir_path: _Optional[str] = ..., lidar_config: _Optional[_Union[LidarConfig, _Mapping]] = ...) -> None: ...
