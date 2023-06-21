# Experiments

## How to make an environment

```
env, sound_list = env_makers['env_name'](env_config)
RewardWrapper(
    ObservationWrapper(
        ActionWrapper(
            env
        )
    )
)

train_xxx (

)
```

# Environment list

```
env_makers = {
    "husk": make_husk_environment,
    "husks": make_husks_environment,
    "husk-noisy": make_husk_noisy_environment,
    "husks-noisy": make_husks_noisy_environment,
    "husk-darkness": make_husk_darkness_environment,
    "husks-darkness": make_husks_darkness_environment,
    "find-animal": make_find_animal_environment,
    "husk-random": make_random_husk_environment,
    "husks-random": make_random_husks_environment,
    "husks-random-darkness": make_random_husks_darkness_environment,
    "husks-continuous": make_continuous_husks_environment,
    "husk-random-terrain": make_random_husk_terrain_environment,
    "husk-hunt": make_hunt_husk_environment,
}
```

| Env name              | Description                                                                   |
|-----------------------|-------------------------------------------------------------------------------|
| husk                  | Escaping from a single husk in a superflat world. The husk position is fixed. |
| husks                 | Escaping from multiple husks in a superflat world. The positions are fixed.   |
| husk-noisy            | Escaping from a husk, with many other animals.                                |
| husks-noisy           | Escaping from husks, with many other animals                                  |
| husk-darkness         | Escaping from a husk, with darkness effect                                    |
| husks-darkness        | Escaping from husks, with darkness effect                                     |
| find-animal           | Searching for randomly arranged animals in a animal pen                       |
| husk-random           | Escaping from a randomly positioned husk.                                     |
| husks-random          | Escaping from randomly positioned husks.                                      |
| husks-random-darkness | Escaping from randomly positioned husks with darkness effect applied          |
| husks-continuous      | Husks are summoned nearby the player continuously                             |
| husk-random-terrain   | Escape from a husk, in a normal terrain                                       |
| husk-hunt             | Hunting a husk in a superflat world using a diamond sword.                    |


