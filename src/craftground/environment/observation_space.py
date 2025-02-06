import gymnasium as gym
import numpy as np
from gymnasium import spaces


def declare_observation_space(image_width: int, image_height: int) -> gym.spaces.Dict:
    entity_info_space = gym.spaces.Dict(
        {
            "unique_name": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,),
                dtype=np.int32,
            ),
            "translation_key": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,),
                dtype=np.int32,
            ),
            "x": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,),
                dtype=np.float64,
            ),
            "y": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,),
                dtype=np.float64,
            ),
            "z": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,),
                dtype=np.float64,
            ),
            "yaw": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,),
                dtype=np.float64,
            ),
            "pitch": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,),
                dtype=np.float64,
            ),
            "health": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,),
                dtype=np.float64,
            ),
        }
    )
    sound_entry_space = gym.spaces.Dict(
        {
            "translate_key": spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.int32
            ),
            "x": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
            "y": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
            "z": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
            "age": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.int32),
        }
    )
    entities_within_distance_space = gym.spaces.Sequence(entity_info_space)
    status_effect_space = gym.spaces.Dict(
        {
            "translation_key": spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.int32
            ),
            "amplifier": spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.int32
            ),
            "duration": spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.int32
            ),
        }
    )
    return gym.spaces.Dict(
        {
            "obs": spaces.Dict(
                {
                    "image": spaces.Box(
                        low=0,
                        high=255,
                        shape=(image_height, image_width, 3),
                        dtype=np.uint8,
                    ),
                    "position": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                    ),
                    "yaw": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64
                    ),
                    "pitch": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64
                    ),
                    "health": spaces.Box(
                        low=0, high=np.inf, shape=(1,), dtype=np.float64
                    ),
                    "food_level": spaces.Box(
                        low=0, high=np.inf, shape=(1,), dtype=np.float64
                    ),
                    "saturation_level": spaces.Box(
                        low=0, high=np.inf, shape=(1,), dtype=np.float64
                    ),
                    "is_dead": spaces.Discrete(2),
                    "inventory": spaces.Sequence(
                        spaces.Dict(
                            {
                                "raw_id": spaces.Box(
                                    low=-np.inf,
                                    high=np.inf,
                                    shape=(1,),
                                    dtype=np.int32,
                                ),
                                "translation_key": spaces.Box(
                                    low=-np.inf,
                                    high=np.inf,
                                    shape=(1,),
                                    dtype=np.int32,
                                ),
                                "count": spaces.Box(
                                    low=-np.inf,
                                    high=np.inf,
                                    shape=(1,),
                                    dtype=np.int32,
                                ),
                                "durability": spaces.Box(
                                    low=-np.inf,
                                    high=np.inf,
                                    shape=(1,),
                                    dtype=np.int32,
                                ),
                                "max_durability": spaces.Box(
                                    low=-np.inf,
                                    high=np.inf,
                                    shape=(1,),
                                    dtype=np.int32,
                                ),
                            }
                        ),
                    ),
                    "raycast_result": spaces.Dict(
                        {
                            "type": spaces.Discrete(3),
                            "target_block": spaces.Dict(
                                {
                                    "x": spaces.Box(
                                        low=-np.inf,
                                        high=np.inf,
                                        shape=(1,),
                                        dtype=np.int32,
                                    ),
                                    "y": spaces.Box(
                                        low=-np.inf,
                                        high=np.inf,
                                        shape=(1,),
                                        dtype=np.int32,
                                    ),
                                    "z": spaces.Box(
                                        low=-np.inf,
                                        high=np.inf,
                                        shape=(1,),
                                        dtype=np.int32,
                                    ),
                                    "translation_key": spaces.Box(
                                        low=-np.inf,
                                        high=np.inf,
                                        shape=(1,),
                                        dtype=np.int32,
                                    ),
                                }
                            ),
                            "target_entity": entity_info_space,
                        }
                    ),
                    "sound_subtitles": spaces.Sequence(sound_entry_space),
                    "status_effects": spaces.Sequence(status_effect_space),
                    "killed_statistics": spaces.Dict(),
                    "mined_statistics": spaces.Dict(),
                    "misc_statistics": spaces.Dict(),
                    "visible_entities": spaces.Sequence(entity_info_space),
                    "surrounding_entities": entities_within_distance_space,  # This is actually
                    "bobber_thrown": spaces.Discrete(2),
                    "experience": spaces.Box(
                        low=0, high=np.inf, shape=(1,), dtype=np.int32
                    ),
                    "world_time": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(1,), dtype=np.int64
                    ),
                    "last_death_message": spaces.Text(min_length=0, max_length=1000),
                    "image_2": spaces.Box(
                        low=0,
                        high=255,
                        shape=(image_height, image_width, 3),
                        dtype=np.uint8,
                    ),
                }
            ),
        }
    )
