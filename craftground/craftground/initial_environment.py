# import pdb
from craftground.craftground.screen_encoding_modes import ScreenEncodingMode


# initial_env = InitialEnvironment(["sword", "shield"], [10, 20], ["summon ", "killMob"], 800, 600, 123456, True, False, False, "sunny")
class InitialEnvironment:
    def __init__(
        self,
        initialInventoryCommands,
        initialPosition,
        initialMobsCommands,
        imageSizeX,
        imageSizeY,
        seed,
        allowMobSpawn,
        alwaysNight,
        alwaysDay,
        initialWeather,
        isHardCore,
        isWorldFlat,
        is_biocular=False,
        visibleSizeX=None,
        visibleSizeY=None,
        initialExtraCommands=None,
        killedStatKeys=None,
        minedStatKeys=None,
        miscStatKeys=None,
        obs_keys=None,
        surrounding_entities_keys=None,
        isHudHidden: bool = False,
        render_distance: int = 5,
        simulation_distance: int = 5,
        eye_distance: float = 0.1,
        structure_paths=None,
        noWeatherCycle=True,
        noTimeCycle=True,
        no_pov_effect=False,
        request_raycast=False,
        screen_encoding_mode: ScreenEncodingMode = ScreenEncodingMode.RAW,
        **kwargs,
    ):
        if structure_paths is None:
            structure_paths = []
        self.initialInventoryCommands = initialInventoryCommands
        self.initialPosition = initialPosition
        self.initialMobsCommands = initialMobsCommands
        self.imageSizeX = imageSizeX
        self.imageSizeY = imageSizeY
        self.seed = seed
        self.allowMobSpawn = allowMobSpawn
        self.alwaysNight = alwaysNight
        self.alwaysDay = alwaysDay
        self.initialWeather = initialWeather
        self.isHardCore = isHardCore
        self.isWorldFlat = isWorldFlat
        self.visibleSizeX = imageSizeX if visibleSizeX is None else visibleSizeX
        self.visibleSizeY = imageSizeY if visibleSizeY is None else visibleSizeY
        self.initialExtraCommands = initialExtraCommands
        self.killedStatKeys = killedStatKeys
        self.minedStatKeys = minedStatKeys
        self.miscStatKeys = miscStatKeys
        self.obs_keys = obs_keys
        self.surrounding_entities_keys = surrounding_entities_keys
        self.isHudHidden = isHudHidden
        self.render_distance = render_distance
        self.simulation_distance = simulation_distance
        self.is_biocular = is_biocular
        self.eye_distance = eye_distance
        self.structure_paths = structure_paths
        self.noWeatherCycle = noWeatherCycle
        self.no_pov_effect = no_pov_effect
        self.noTimeCycle = noTimeCycle
        self.request_raycast = request_raycast
        self.screen_encoding_mode = screen_encoding_mode
        if kwargs:
            print(f"Unexpected Kwargs: {kwargs}")

        if self.visibleSizeX != self.imageSizeX or self.visibleSizeY != self.imageSizeY:
            raise ValueError(
                "Visible Size is deprecated. Visible size must be equal to image size."
            )
        if self.alwaysNight and self.alwaysDay:
            raise ValueError("Cannot have both alwaysNight and alwaysDay set to true.")
