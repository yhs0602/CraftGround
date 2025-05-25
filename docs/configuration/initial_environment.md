---
title: Initial Environment Configuration
parent: Configuration
---

# Initial Environment Configuration

## `InitialEnvironmentConfig`  

The `InitialEnvironmentConfig` class defines the configuration parameters for initializing a Minecraft environment. It provides options for setting up rendering properties, game settings, world generation, and additional features needed for reinforcement learning or simulation tasks.

### **Constructor Parameters**  

| Parameter                      | Type                 | Default                  | Description                                                                                                                                                                |
| ------------------------------ | -------------------- | ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `image_width`                  | `int`                | `640`                    | Width of the rendered image.                                                                                                                                               |
| `image_height`                 | `int`                | `360`                    | Height of the rendered image.                                                                                                                                              |
| `gamemode`                     | `GameMode`           | `GameMode.SURVIVAL`      | Game mode (Survival, Hardcore, Creative).                                                                                                                                  |
| `difficulty`                   | `Difficulty`         | `Difficulty.NORMAL`      | Difficulty level (Peaceful, Easy, Normal, Hard).                                                                                                                           |
| `world_type`                   | `WorldType`          | `WorldType.DEFAULT`      | Type of world generation (Default, Superflat, Large Biomes, Amplified, Single Biome).                                                                                      |
| `world_type_args`              | `str`                | `""`                     | Additional arguments for world type settings.                                                                                                                              |
| `seed`                         | `str`                | `""`                     | World seed for deterministic world generation.                                                                                                                             |
| `generate_structures`          | `bool`               | `True`                   | Whether to generate structures like villages and dungeons.                                                                                                                 |
| `bonus_chest`                  | `bool`               | `False`                  | Whether to spawn a bonus chest at the start.                                                                                                                               |
| `datapack_paths`               | `List[str]`          | `None`                   | Paths to custom datapacks to load.                                                                                                                                         |
| `initial_extra_commands`       | `List[str]`          | `None`                   | Commands executed at world initialization.                                                                                                                                 |
| `killed_stat_keys`             | `List[str]`          | `None`                   | Statistics keys related to player kills.                                                                                                                                   |
| `mined_stat_keys`              | `List[str]`          | `None`                   | Statistics keys related to blocks mined.                                                                                                                                   |
| `misc_stat_keys`               | `List[str]`          | `None`                   | Other relevant player statistics keys.                                                                                                                                     |
| `surrounding_entity_distances` | `List[int]`          | `None`                   | Distances for tracking surrounding entities.                                                                                                                               |
| `hud_hidden`                   | `bool`               | `False`                  | Whether to hide the in-game HUD.                                                                                                                                           |
| `render_distance`              | `int`                | `6`                      | Number of chunks rendered around the player.                                                                                                                               |
| `simulation_distance`          | `int`                | `6`                      | Number of chunks simulated around the player.                                                                                                                              |
| `eye_distance`                 | `float`              | `0.0`                    | Distance between the eyes (e.g., for binocular vision).                                                                                                                    |
| `structure_paths`              | `List[str]`          | `None`                   | Paths to pre-defined structures to be loaded.                                                                                                                              |
| `no_fov_effect`                | `bool`               | `False`                  | Disables FOV changes due to movement effects.                                                                                                                              |
| `request_raycast`              | `bool`               | `False`                  | Whether to request raycasting data.                                                                                                                                        |
| `screen_encoding_mode`         | `ScreenEncodingMode` | `ScreenEncodingMode.RAW` | Encoding mode for screen output.                                                                                                                                           |
| `requires_surrounding_blocks`  | `bool`               | `False`                  | Whether to include surrounding block information.                                                                                                                          |
| `level_display_name_to_play`   | `str`                | `""`                     | Name of the level to load.                                                                                                                                                 |
| `fov`                          | `int`                | `70`                     | Field of view setting.                                                                                                                                                     |
| `requires_biome_info`          | `bool`               | `False`                  | Whether to include biome information in observations.                                                                                                                      |
| `requires_heightmap`           | `bool`               | `False`                  | Whether to include the world's heightmap.                                                                                                                                  |
| `requires_depth`               | `bool`               | `False`                  | Whether to include depth information.                                                                                                                                      |
| `requires_depth_conversion`    | `bool`               | `True`                   | If `True`, depth values are transformed into real-world distances and normalized. The conversion uses a near plane of `0.05` and a far plane set to `4.0 * view distance`. |
| `resource_zip_path`            | `str`                | `""`                     | Path to a resource pack zip file.                                                                                                                                          |
| `block_collision_keys`         | `List[str]`          | `None`                   | Keys for block collision detection.                                                                                                                                        |
| `entity_collision_keys`        | `List[str]`          | `None`                   | Keys for entity collision detection.                                                                                                                                       |


### **GameMode Enum**
Defines the available game modes.

| Value      | Description                                  |
| ---------- | -------------------------------------------- |
| `SURVIVAL` | Standard gameplay with survival mechanics.   |
| `HARDCORE` | Like survival, but with perma-death enabled. |
| `CREATIVE` | Free building mode with unlimited resources. |

### **Difficulty Enum**
Defines the difficulty settings.

| Value      | Description                                            |
| ---------- | ------------------------------------------------------ |
| `PEACEFUL` | No hostile mobs, health regenerates automatically.     |
| `EASY`     | Hostile mobs spawn but deal reduced damage.            |
| `NORMAL`   | Standard difficulty.                                   |
| `HARD`     | More difficult survival experience with stronger mobs. |

### **WorldType Enum**
Defines different world generation types.

| Value          | Description                         |
| -------------- | ----------------------------------- |
| `DEFAULT`      | Standard Minecraft terrain.         |
| `SUPERFLAT`    | A completely flat world.            |
| `LARGE_BIOMES` | Biomes are larger than normal.      |
| `AMPLIFIED`    | Extremely tall and chaotic terrain. |
| `SINGLE_BIOME` | Entire world consists of one biome. |

### **DaylightMode Enum**
Defines time-related settings.

| Value          | Description                          |
| -------------- | ------------------------------------ |
| `DEFAULT`      | Normal day/night cycle.              |
| `ALWAYS_DAY`   | Locked to daytime.                   |
| `ALWAYS_NIGHT` | Locked to nighttime.                 |
| `FREEZED`      | Time is frozen at a specific moment. |

---

## **Related Data Structures**

### **BlockState**
Defines the state of a specific block in the environment.

| Field         | Type     | Description                                                               |
| ------------- | -------- | ------------------------------------------------------------------------- |
| `x, y, z`     | `int32`  | Block's coordinates.                                                      |
| `block_state` | `string` | Minecraft block state string, e.g., `minecraft:stone[waterlogged=false]`. |

### **InitialEnvironmentMessage**
Defines the initial state of the environment.

| Field                                         | Type                  | Description                                     |
| --------------------------------------------- | --------------------- | ----------------------------------------------- |
| `initialInventoryCommands`                    | `repeated string`     | Commands for setting up the player's inventory. |
| `initialPosition`                             | `repeated int32`      | Starting coordinates of the player.             |
| `initialMobsCommands`                         | `repeated string`     | Commands to spawn mobs at initialization.       |
| `imageSizeX, imageSizeY`                      | `int32`               | Resolution of the environment capture.          |
| `seed`                                        | `int64`               | World generation seed.                          |
| `allowMobSpawn`                               | `bool`                | Whether mobs can spawn.                         |
| `alwaysNight, alwaysDay`                      | `bool`                | Time cycle settings.                            |
| `initialWeather`                              | `string`              | Starting weather condition.                     |
| `isWorldFlat`                                 | `bool`                | Whether the world is a flat world.              |
| `visibleSizeX, visibleSizeY`                  | `int32`               | Visible screen size of the player.              |
| `initialExtraCommands`                        | `repeated string`     | Extra initialization commands.                  |
| `killedStatKeys, minedStatKeys, miscStatKeys` | `repeated string`     | Keys for tracking player statistics.            |
| `initialBlockStates`                          | `repeated BlockState` | Initial block configurations.                   |
| `surroundingEntityDistances`                  | `repeated int32`      | Tracking distances for entities.                |
| `hudHidden`                                   | `bool`                | Whether the HUD is hidden.                      |
| `render_distance, simulation_distance`        | `int32`               | View and simulation distance settings.          |
| `python_pid`                                  | `int32`               | Process ID of the controlling Python script.    |
| `requiresDepth`                               | `bool`                | Whether depth information is required.          |
| `requiresDepthConversion`                     | `bool`                | Whether depth data requires conversion.         |
| `resourceZipPath`                             | `str`                 | Path to a resource pack zip file.               |
| `blockCollisionKeys`                          | `List[str]`           | Keys for block collision detection.             |
| `entityCollisionKeys`                         | `List[str]`           | Keys for entity collision detection.            |
