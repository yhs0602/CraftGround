## Observation Space

### `ItemStack`

| Field                      | Type   | Description                          |
|----------------------------|--------|--------------------------------------|
| raw_id                     | int32  | Unique item identifier.              |
| translation_key            | string | Item's display name translation key. |
| count                      | int32  | Amount in the item stack.            |
| durability, max_durability | int32  | Durability information of item.      |

### `BlockInfo`

| Field           | Type   | Description                           |
|-----------------|--------|---------------------------------------|
| x, y, z         | int32  | Block coordinates.                    |
| translation_key | string | Block's display name translation key. |

### `EntityInfo`

| Field           | Type   | Description                  |
|-----------------|--------|------------------------------|
| unique_name     | string | Unique name of the entity.   |
| translation_key | string | Entity's translation key.    |
| x, y, z         | double | Entity coordinates.          |
| yaw, pitch      | double | Yaw and Pitch of the entity. |
| health          | double | Health of the entity.        |

### `ObservationSpaceMessage`

| Field                | Type                               | Description                                                |
|----------------------|------------------------------------|------------------------------------------------------------|
| image                | bytes                              | Image data of the environment.                             |
| x, y, z              | double                             | Player's coordinates in the world.                         |
| yaw, pitch           | double                             | Player's orientation (yaw & pitch).                        |
| health               | double                             | Player's health level.                                     |
| food_level           | double                             | Player's food level.                                       |
| saturation_level     | double                             | Player's saturation level.                                 |
| is_dead              | bool                               | Flag indicating if the player is dead.                     |
| inventory            | repeated ItemStack                 | List of items in player's inventory.                       |
| raycast_result       | HitResult                          | Raycasting result to identify targeted blocks or entities. |
| sound_subtitles      | repeated SoundEntry                | List of recent sounds with subtitles.                      |
| status_effects       | repeated StatusEffect              | List of active status effects on the player.               |
| killed_statistics    | map<string, int32>                 | Player's kill statistics with entity names as keys.        |
| mined_statistics     | map<string, int32>                 | Player's block mining statistics with block types as keys. |
| misc_statistics      | map<string, int32>                 | Miscellaneous statistics.                                  |
| visible_entities     | repeated EntityInfo                | List of entities currently visible to the player.          |
| surrounding_entities | map<int32, EntitiesWithinDistance> | Map of entities around the player with distances as keys.  |
| bobber_thrown        | bool                               | Flag indicating if a fishing bobber is currently thrown.   |
| world_time           | int64                              | Current world time.                                        |
| last_death_message   | string                             | Last death reason.                                         |
