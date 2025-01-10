## Initial Environment

### `BlockState`

| Field       | Type   | Description                                                                                                      |
|-------------|--------|------------------------------------------------------------------------------------------------------------------|
| x, y, z     | int32  | Coordinates of the block.                                                                                        |
| block_state | string | State of the block, e.g., `minecraft:andesite_stairs[facing=east,half=bottom,shape=straight,waterlogged=false]`. |

### `InitialEnvironmentMessage`

| Field                                       | Type                | Description                                   |
|---------------------------------------------|---------------------|-----------------------------------------------|
| initialInventoryCommands                    | repeated string     | Commands to setup initial inventory.          |
| initialPosition                             | repeated int32      | Player's starting coordinates.                |
| initialMobsCommands                         | repeated string     | Commands for spawning mobs.                   |
| imageSizeX, imageSizeY                      | int32               | Image dimensions of the environment.          |
| seed                                        | int64               | World generation seed.                        |
| allowMobSpawn                               | bool                | Controls mob spawning.                        |
| alwaysNight, alwaysDay                      | bool                | Control for time of day.                      |
| initialWeather                              | string              | Initial weather setting.                      |
| isWorldFlat                                 | bool                | Flat world toggle.                            |
| visibleSizeX, visibleSizeY                  | int32               | Player's visible dimensions.                  |
| initialExtraCommands                        | repeated string     | Extra commands for initialization.            |
| killedStatKeys, minedStatKeys, miscStatKeys | repeated string     | Player's statistic keys.                      |
| initialBlockStates                          | repeated BlockState | Initial world block states.                   |
| surroundingEntityDistances                  | repeated int32      | Entity distances from player.                 |
| hudHidden                                   | bool                | Toggle for HUD visibility.                    |
| render_distance, simulation_distance        | int32               | Block and entity render/simulation distances. |