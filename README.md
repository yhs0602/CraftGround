# CraftGround - Reinforcement Learning Environment for Minecraft
[![Wheels](https://github.com/yhs0602/CraftGround/actions/workflows/publish-package.yml/badge.svg)](https://github.com/yhs0602/CraftGround/actions/workflows/publish-package.yml)
[![Python package](https://github.com/yhs0602/CraftGround/actions/workflows/python-ci.yml/badge.svg)](https://github.com/yhs0602/CraftGround/actions/workflows/python-ci.yml)
[![CMake Build](https://github.com/yhs0602/CraftGround/actions/workflows/cmake-build.yml/badge.svg)](https://github.com/yhs0602/CraftGround/actions/workflows/cmake-build.yml)
[![Gradle Build](https://github.com/yhs0602/CraftGround/actions/workflows/gradle.yml/badge.svg)](https://github.com/yhs0602/CraftGround/actions/workflows/gradle.yml)

<img src="docs/craftground.webp" alt="CraftGround_Logo" width="50%"/>


[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fyhs0602%2FMinecraftRL)](https://github.com/yhs0602/MinecraftRL)

RL experiments using lightweight minecraft environment

This is the latest development repository of CraftGround environment.

## Quick start

1. You need to install JDK >= 21
1. Run the following command to install the package.
    ```shell
    pip install craftground
    ```
1. Take a look at the [the demo repository](https://github.com/yhs0602/CraftGround-Baselines3)!
1. Here is a simple example that uses this environment.
    ```python
   from craftground import craftground
   from craftground.wrappers.action import ActionWrapper, Action
   from craftground.wrappers.fast_reset import FastResetWrapper
   from craftground.wrappers.time_limit import TimeLimitWrapper
   from craftground.wrappers.vision import VisionWrapper
   from stable_baselines3 import A2C
   from stable_baselines3.common.monitor import Monitor
   from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
   from wandb.integration.sb3 import WandbCallback
   
   import wandb
   from avoid_damage import AvoidDamageWrapper
   
   
   def main():
       run = wandb.init(
           # set the wandb project where this run will be logged
           project="craftground-sb3",
           entity="jourhyang123",
           # track hyperparameters and run metadata
           group="escape-husk",
           sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
           monitor_gym=True,  # auto-upload the videos of agents playing the game
           save_code=True,  # optional    save_code=True,  # optional
       )
       env = craftground.make(
           # env_path="../minecraft_env",
           port=8023,
           initialInventoryCommands=[],
           initialPosition=None,  # nullable
           initialMobsCommands=[
               "minecraft:husk ~ ~ ~5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
               # player looks at south (positive Z) when spawn
           ],
           imageSizeX=114,
           imageSizeY=64,
           visibleSizeX=114,
           visibleSizeY=64,
           seed=12345,  # nullable
           allowMobSpawn=False,
           alwaysDay=True,
           alwaysNight=False,
           initialWeather="clear",  # nullable
           isHardCore=False,
           isWorldFlat=True,  # superflat world
           obs_keys=["sound_subtitles"],
           initialExtraCommands=[],
           isHudHidden=False,
           render_action=True,
           render_distance=2,
           simulation_distance=5,
       )
       env = FastResetWrapper(
           TimeLimitWrapper(
               ActionWrapper(
                   AvoidDamageWrapper(VisionWrapper(env, x_dim=114, y_dim=64)),
                   enabled_actions=[Action.FORWARD, Action.BACKWARD],
               ),
               max_timesteps=400,
           )
       )
       env = Monitor(env)
       env = DummyVecEnv([lambda: env])
       env = VecVideoRecorder(
           env,
           f"videos/{run.id}",
           record_video_trigger=lambda x: x % 2000 == 0,
           video_length=200,
       )
       model = A2C(
           "MlpPolicy", env, verbose=1, device="mps", tensorboard_log=f"runs/{run.id}"
       )
   
       model.learn(
           total_timesteps=10000,
           callback=WandbCallback(
               gradient_save_freq=100,
               model_save_path=f"models/{run.id}",
               verbose=2,
           ),
       )
       model.save("a2c_craftground")
       run.finish()
   
       # vec_env = model.get_env()
       # obs = vec_env.reset()
       # for i in range(1000):
       #     action, _state = model.predict(obs, deterministic=True)
       #     obs, reward, done, info = vec_env.step(action)
       #     # vec_env.render("human")
       #     # VecEnv resets automatically
       #     # if done:
       #     #   obs = vec_env.reset()
   
   
   if __name__ == "__main__":
       main()

    ```

# Environment

https://github.com/yhs0602/MinecraftEnv

Utilizing protocol buffers, we've constructed a reinforcement learning environment specifically tailored for Minecraft.
Below are detailed specifications of the environment's architecture:

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

## Action Space

### `ActionSpaceMessage`

| Field    | Type            | Description               |
|----------|-----------------|---------------------------|
| action   | repeated int32  | Available player actions. |
| commands | repeated string | Any minecraft commands.   |


# Wrapper list

| Wrapper Name            | Description                                                                                         |
|-------------------------|-----------------------------------------------------------------------------------------------------|
| action                  | Defines discrete action spaces and operations for the agent.                                        |
| sound                   | Provides sound-based feedback or actions for the agent.                                             |
| vision                  | Incorporates visual feedback or vision-based actions for the agent.                                 |

# How to execute minecraft command in a gymnasium wrapper?
```python
self.get_wrapper_attr("add_commands")(
    [
        f"setblock 1 2 3 minecraft:cake"
    ]
)
```


# Devaju font license

https://dejavu-fonts.github.io/License.html


# Adding a new paramerter
1. Edit .proto
2. protoc
3. Edit python files


# Formatting
## Install formatters
```zsh
brew install ktlint clang-format google-java-format
```
```bash
find . \( -iname '*.h' -o -iname '*.cpp' -o -iname '*.mm' \) | xargs clang-format -i
ktlint '!src/craftground/MinecraftEnv/src/main/java/com/kyhsgeekcode/minecraftenv/proto/**'
find . -name '*.java' -print0 | xargs -0 -P 4 google-java-format -i
```

# Managing proto files
```bash
cd src/
protoc proto/action_space.proto --python_out=craftground
protoc proto/initial_environment.proto --python_out=craftground
protoc proto/observation_space.proto --python_out=craftground
protoc proto/action_space.proto --java_out=craftground/MinecraftEnv/src/main/java/ --kotlin_out=craftground/MinecraftEnv/src/main/java/
protoc proto/initial_environment.proto --java_out=craftground/MinecraftEnv/src/main/java/ --kotlin_out=craftground/MinecraftEnv/src/main/java/
protoc proto/observation_space.proto --java_out=craftground/MinecraftEnv/src/main/java/ --kotlin_out=craftground/MinecraftEnv/src/main/java/
```