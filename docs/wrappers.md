# Wrappers
Craftground provides a set of wrappers that can be used to easily select the observation and action spaces for the agent. These wrappers can be used to define the agent's capabilities and the environment's feedback mechanisms. The following table lists the available wrappers and their descriptions.

| Wrapper Name  | Description                                                         |
| ------------- | ------------------------------------------------------------------- |
| ActionWrapper | Defines discrete action spaces and operations for the agent.        |
| SoundWrapper  | Provides sound-based feedback or actions for the agent.             |
| VisionWrapper | Incorporates visual feedback or vision-based actions for the agent. |


## ActionWrapper
ActionWrapper defines discrete action spaces and operations for the agent. The wrapper provides a set of actions that the agent can perform in the environment. The following table lists the available actions and their descriptions.

| Action Name  | Description                                   |
| ------------ | --------------------------------------------- |
| NO_OP        | No operation.                                 |
| FORWARD      | Move the agent forward.                       |
| BACKWARD     | Move the agent backward.                      |
| STRAFE_RIGHT | Move the agent to the right.                  |
| STRAFE_LEFT  | Move the agent to the left.                   |
| JUMP         | Make the agent jump.                          |
| LOOK_UP      | Change the agent's camera pitch to look up.   |
| LOOK_DOWN    | Change the agent's camera pitch to look down. |
| ATTACK       | Attack the entity in front of the agent.      |
| USE          | Use the item in the agent's hand.             |
| JUMP_USE     | Drop the item in the agent's hand.            |

ActionWrapper implementation shows how to convert CraftGround's dictionary action space to a discrete action space.

### Action Space
```
Discrete(11)
```

## VisionWrapper
VisionWrapper selects only the visual information from the observation space.

### Observation Space
```
Box(0, 255, (y_dim, x_dim, 3), uint8)
```

## SoundWrapper
SoundWrapper selects only the sound information from the observation space.

```python
class SoundWrapper(gymnasium.Wrapper):
    def __init__(
        self, env, sound_list, zeroing_sound_list, coord_dim, null_value=0.0, **kwargs
    ):
```

- `env`: Environment to be wrapped.
- `sound_list`: List of sounds to be considered.
- `zeroing_sound_list`: List of sounds that happens at the position of the agent.
- `coord_dim`: Number of coordinates for each sound.
- `null_value`: Value to be filled for the sounds that are not in the observation space.


### Observation Space
```
Box(-1, 1, (num_sounds * coodinate_dim + zeroing_num_sounds + 2), float32)
```
- `num_sounds`: Number of sounds to be considered.
- `coodinate_dim`: Number of coordinates for each sound.
- `zeroing_num_sounds`: Number of sounds that happens at the position of the agent.
- `2`: Cos and Sin value of agent's yaw.

The values of the sounds are the normalized value of distance between the sound source and the agent divided by 15, resulting in between -1 and 1. Note that the sounds played over 15 blocks away from the agent are not considered in the observation space. 

Zeroing sounds are the sounds that happen at the position of the agent. The value of the sound is set to 1 if the sound is played at the agent. The agent's yaw is also included in the observation space as Cos and Sin values.

### Example
For example, if you want to consider the sounds of `minecraft:entity.zombie.ambient`, `minecraft:entity.zombie.death`, and `minecraft:entity.zombie.hurt`, and the zeroing sound of `subtitles.entity.player.hurt`, you can use the following code snippet.

```python
env = SoundWrapper(
    env,
    sound_list=[
        "minecraft:entity.zombie.ambient",
        "minecraft:entity.zombie.death",
        "minecraft:entity.zombie.hurt",
    ],
    zeroing_sound_list=["subtitles.entity.player.hurt"],
    coord_dim=3,
)
```

Then an example observation values will be as follows:

```
[
    0.2, 0.2, 0.2 # zombie.ambient
    0.5, 0.5, 0.5 # zombie.death
    0.8, 0.8, 0.8 # zombie.hurt
    1.0 # player.hurt
    0.5, 0.87 # Cos and Sin of agent's yaw
]
```

