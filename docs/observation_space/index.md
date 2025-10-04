---
title: Observation Space
nav_order: 4
---

# **Observation Space**

The observation space represents the agent's perception of the Minecraft environment, including visual input, player state, entities, blocks, and other relevant data.

## **Usage of Observation Space**

The observation space is returned as a dictionary containing structured information about the environment.  

```python
obs, info = env.reset()
obs: Dict[str, Union[np.ndarray, "torch.Tensor"]]
obs["full"]: ObservationSpaceMessage  # Full structured observation data
obs["pov"]: np.ndarray | "torch.Tensor"  # Visual observation (RGB image)
```

- `obs["full"]`: Contains the complete structured observation data (`ObservationSpaceMessage`).
- `obs["pov"]`: Contains the primary visual observation as an RGB image (NumPy array or PyTorch tensor, depending on the `screen_encoding_mode` used).

---

## **ObservationSpaceMessage**
The main observation structure, containing all relevant environment information.

| Field                  | Type                                 | Description                                                 |
| ---------------------- | ------------------------------------ | ----------------------------------------------------------- |
| `image`                | `bytes`                              | Primary visual observation (RGB image).                     |
| `x, y, z`              | `double`                             | Agent's position in the world.                              |
| `yaw, pitch`           | `double`                             | Agent's orientation.                                        |
| `health`               | `double`                             | Agent's health level.                                       |
| `food_level`           | `double`                             | Agent's hunger level.                                       |
| `saturation_level`     | `double`                             | Agent's saturation level.                                   |
| `is_dead`              | `bool`                               | Whether the agent is dead.                                  |
| `inventory`            | `repeated ItemStack`                 | Items in the player's inventory.                            |
| `raycast_result`       | `HitResult`                          | Information about the block/entity the agent is looking at. |
| `sound_subtitles`      | `repeated SoundEntry`                | List of detected environmental sounds.                      |
| `status_effects`       | `repeated StatusEffect`              | List of active status effects.                              |
| `killed_statistics`    | `map<string, int32>`                 | Map of killed entities and their counts.                    |
| `mined_statistics`     | `map<string, int32>`                 | Map of mined blocks and their counts.                       |
| `misc_statistics`      | `map<string, int32>`                 | Miscellaneous statistics.                                   |
| `visible_entities`     | `repeated EntityInfo`                | List of entities currently visible to the player.           |
| `surrounding_entities` | `map<int32, EntitiesWithinDistance>` | Entities within specified distances.                        |
| `bobber_thrown`        | `bool`                               | Whether a fishing bobber is currently in use.               |
| `experience`           | `int32`                              | Agent’s experience points.                                  |
| `world_time`           | `int64`                              | Current in-game time.                                       |
| `last_death_message`   | `string`                             | Message displayed when the agent last died.                 |
| `image_2`              | `bytes`                              | Secondary visual observation (if applicable).               |
| `surrounding_blocks`   | `repeated BlockInfo`                 | Blocks surrounding the agent (3x3x3 grid).                  |
| `eye_in_block`         | `bool`                               | Whether the agent’s camera is inside a block.               |
| `suffocating`          | `bool`                               | Whether the agent is taking suffocation damage.             |
| `chat_messages`        | `repeated ChatMessageInfo`           | Chat messages received.                                     |
| `biome_info`           | `BiomeInfo`                          | Current biome information.                                  |
| `nearby_biomes`        | `repeated NearbyBiome`               | List of nearby biomes.                                      |
| `submerged_in_water`   | `bool`                               | Whether the agent is fully underwater.                      |
| `is_in_lava`           | `bool`                               | Whether the agent is touching lava.                         |
| `submerged_in_lava`    | `bool`                               | Whether the agent is fully submerged in lava.               |
| `height_info`          | `repeated HeightInfo`                | Heightmap data of the surrounding terrain.                  |
| `is_on_ground`         | `bool`                               | Whether the agent is standing on solid ground.              |
| `is_touching_water`    | `bool`                               | Whether the agent is in contact with water.                 |
| `ipc_handle`           | `bytes`                              | Handle for inter-process communication.                     |
| `depth`                | `repeated float [packed=true]`       | Depth buffer information.                                   |
| `block_collisions`     | `repeated BlockCollisionInfo`        | List of block collisions detected by the agent.             |
| `entity_collisions`    | `repeated EntityCollisionInfo`       | List of entity collisions detected by the agent.            |
| `velocity_x`           | `double`                             | Agent's velocity in the X direction.                        |
| `velocity_y`           | `double`                             | Agent's velocity in the Y direction.                        |
| `velocity_z`           | `double`                             | Agent's velocity in the Z direction.                        |


---

## **ItemStack**
Represents an item in the player's inventory.

| Field             | Type     | Description                          |
| ----------------- | -------- | ------------------------------------ |
| `raw_id`          | `int32`  | Unique identifier for the item.      |
| `translation_key` | `string` | Item's display name translation key. |
| `count`           | `int32`  | Number of items in the stack.        |
| `durability`      | `int32`  | Current durability of the item.      |
| `max_durability`  | `int32`  | Maximum durability of the item.      |

---

## **BlockInfo**
Represents a block in the world.

| Field             | Type     | Description                           |
| ----------------- | -------- | ------------------------------------- |
| `x, y, z`         | `int32`  | Block's coordinates.                  |
| `translation_key` | `string` | Block's display name translation key. |

---

## **EntityInfo**
Represents an entity (mobs, players, etc.) in the world.

| Field             | Type     | Description                                                 |
| ----------------- | -------- | ----------------------------------------------------------- |
| `unique_name`     | `string` | Unique entity identifier.                                   |
| `translation_key` | `string` | Entity's display name translation key.                      |
| `x, y, z`         | `double` | Entity's coordinates.                                       |
| `yaw, pitch`      | `double` | Entity's rotation.                                          |
| `health`          | `double` | Entity's health points.                                     |
| `in_love`         | `bool`   | Whether the animal entity is in "love" mode (for breeding). |

---

## **HitResult**
Represents the result of a raycasting operation.

| Field           | Type         | Description                                    |
| --------------- | ------------ | ---------------------------------------------- |
| `type`          | `Type`       | Type of hit (`MISS`, `BLOCK`, `ENTITY`).       |
| `target_block`  | `BlockInfo`  | The block hit by the raycast (if applicable).  |
| `target_entity` | `EntityInfo` | The entity hit by the raycast (if applicable). |

---

## **StatusEffect**
Represents an active status effect on the player.

| Field             | Type     | Description                         |
| ----------------- | -------- | ----------------------------------- |
| `translation_key` | `string` | Status effect name translation key. |
| `duration`        | `int32`  | Duration of the effect in ticks.    |
| `amplifier`       | `int32`  | Strength level of the effect.       |

---

## **SoundEntry**
Represents a sound event detected by the agent.

| Field           | Type     | Description                    |
| --------------- | -------- | ------------------------------ |
| `translate_key` | `string` | Sound translation key.         |
| `age`           | `int64`  | Time since the sound occurred. |
| `x, y, z`       | `double` | Location of the sound source.  |

---

## **EntitiesWithinDistance**
Represents entities within a certain distance from the agent.

| Field      | Type                  | Description              |
| ---------- | --------------------- | ------------------------ |
| `entities` | `repeated EntityInfo` | List of nearby entities. |

---

## **ChatMessageInfo**
Represents a chat message received by the agent.

| Field        | Type     | Description                                   |
| ------------ | -------- | --------------------------------------------- |
| `added_time` | `int64`  | Timestamp when the message was received.      |
| `message`    | `string` | The chat message content.                     |
| `indicator`  | `string` | Indicator for the message (currently unused). |

---

## **BiomeInfo**
Represents the biome the agent is currently in.

| Field        | Type     | Description                         |
| ------------ | -------- | ----------------------------------- |
| `biome_name` | `string` | Name of the biome.                  |
| `center_x`   | `int32`  | X coordinate of the biome's center. |
| `center_y`   | `int32`  | Y coordinate of the biome's center. |
| `center_z`   | `int32`  | Z coordinate of the biome's center. |

---

## **NearbyBiome**
Represents information about nearby biomes.

| Field        | Type     | Description                          |
| ------------ | -------- | ------------------------------------ |
| `biome_name` | `string` | Name of the biome.                   |
| `x, y, z`    | `int32`  | Coordinates of the biome's location. |

---

## **HeightInfo**
Represents height information at a specific location.

| Field        | Type     | Description                       |
| ------------ | -------- | --------------------------------- |
| `x, z`       | `int32`  | Coordinates of the location.      |
| `height`     | `int32`  | Height value at this position.    |
| `block_name` | `string` | Name of the block at this height. |

## BlockCollisionInfo
Represents a block collision detected by the agent.
| Field        | Type     | Description                                  |
| ------------ | -------- | -------------------------------------------- |
| `x, y, z`    | `int32`  | Coordinates of the block collision.          |
| `block_name` | `string` | Name of the block involved in the collision. |

## EntityCollisionInfo
Represents an entity collision detected by the agent.
| Field         | Type     | Description                                   |
| ------------- | -------- | --------------------------------------------- |
| `x, y, z`     | `float`  | Coordinates of the entity collision.          |
| `entity_name` | `string` | Name of the entity involved in the collision. |

