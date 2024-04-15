# Custom Structures in CraftGround

Utilizing custom structures are of significant interest for researchers as they allow for a variety of powerful experiments. In this reinforcement learning environment, unlike other environments, it is easy to experiment with custom structures. This section describes how to create and use custom structures within the CraftGround environment.

## Creating Custom Structures in Minecraft

1. Build your desired structure.
2. Use commands to extract a jigsaw block to your inventory.
`give @p minecraft:jigsaw`
3. Place the jigsaw block at the starting and ending points of your structure, ensuring both have the same name.
4. In a new jigsaw block interface, enter the name of your structure and hit the save button.
5. Finally, copy the structure file from `saves/generated/structures/<structure_name>.nbt`.

For a more detailed tutorial on using jigsaw blocks effectively, refer to [this guide](https://gist.github.com/GentlemanRevvnar/98a8f191f46d28f63592672022c41497).

## Installing Custom Structures in the Environment

When setting up your CraftGround environment, use the `craftground.make` function with the `structure_paths` parameter as follows:

```python
craftground.make(structure_paths=[
    'structure_nbt_path',
])
```
This configuration automatically copies and prepares the structure. Similarly, you can use the initialExtraCommands parameter to set up the environment with initial commands like:

```python
initialExtraCommands=[
    "time set noon",
    "place template minecraft:hmaze1_colored 0 0 0",
    "tp @p 3 1 1 -90 0",
    "effect give @p minecraft:speed infinite 2 true",  # speed effect, particle hidden
]
```

This setup will ensure that your custom structures are ready and in place as soon as you start the simulation.


To sum up, an example is:

```python
import os.path

import wandb
from craftground import craftground
from craftground.wrappers.action import ActionWrapper, Action
from craftground.wrappers.fast_reset import FastResetWrapper
from craftground.wrappers.vision import VisionWrapper
from craftground.craftground.screen_encoding_modes import ScreenEncodingMode
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from wandb.integration.sb3 import WandbCallback


def structure_any():
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="craftground-sb3",
        entity="jourhyang123",
        # track hyperparameters and run metadata
        group="structure-a2c-vision",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional    save_code=True,  # optional
    )
    size_x = 114
    size_y = 64
    base_env, sound_list = (
        craftground.make(
            port=8001,
            initialInventoryCommands=[],
            verbose=True,
            initialPosition=[5, 5, 5],  # nullable
            initialMobsCommands=[],
            imageSizeX=size_x,
            imageSizeY=size_y,
            visibleSizeX=size_x,
            visibleSizeY=size_y,
            seed=12345,  # nullable
            allowMobSpawn=False,
            alwaysDay=True,
            alwaysNight=False,
            initialWeather="clear",  # nullable
            isHardCore=False,
            isWorldFlat=True,  # superflat world
            obs_keys=[],  # No sound subtitles
            miscStatKeys=[],  # No stats
            initialExtraCommands=[
                "place template minecraft:portable_maze 0 0 0",
                "tp @p 5 5 5 -90 0",
            ],  # x y z yaw pitch
            isHudHidden=True,
            render_action=True,
            render_distance=5,
            simulation_distance=5,
            structure_paths=[
                os.path.abspath("portable_maze.nbt"),
            ],
            no_pov_effect=True,
            screen_encoding_mode=ScreenEncodingMode.RAW,
            use_vglrun=False, # check_vglrun(),
        ),
        [],
    )
    env = FastResetWrapper(
        ActionWrapper(
            VisionWrapper(
                base_env,
                x_dim=size_x,
                y_dim=size_y,
            ),
            enabled_actions=[
                Action.FORWARD,
                Action.TURN_LEFT,
                Action.TURN_RIGHT,
            ],
        ),
    )
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 4000 == 0,
        video_length=400,
    )
    model = A2C(
        "CnnPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}"
    )

    model.learn(
        total_timesteps=1000,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    # model.save("dqn_sound_husk")
    run.finish()
    base_env.terminate()


if __name__ == "__main__":
    structure_any()



```