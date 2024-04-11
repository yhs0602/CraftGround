# Integrating with Stable Baselines3
You can integrate [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/) with this project.
Here is an example of how to do it:

```python
import os.path

import wandb
from craftground import craftground
from craftground.wrappers.action import ActionWrapper, Action
from craftground.wrappers.fast_reset import FastResetWrapper
from craftground.wrappers.time_limit import TimeLimitWrapper
from craftground.wrappers.vision import VisionWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from wandb.integration.sb3 import WandbCallback
from craftground.craftground.screen_encoding_modes import ScreenEncodingMode
from check_vglrun import check_vglrun

from get_device import get_device

current_path = os.path.dirname(os.path.abspath(__file__))
map_path = os.path.join(current_path, "custom_structure.nbt")

def example_exploration():
    group_name = "example-exploration"
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="craftground-sb3",
        entity="your-name",
        # track hyperparameters and run metadata
        group=group_name,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    size_x = 114
    size_y = 64
    base_env, sound_list = (
        craftground.make(
            port=8001,
            initialInventoryCommands=[],
            verbose=False,
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
                "time set noon",
                "place template minecraft:custom_structure 0 0 0",
                "tp @p 3 1 1 -90 0",
            ],  # x y z yaw pitch
            isHudHidden=True,
            render_action=False,
            render_distance=5,
            simulation_distance=5,
            structure_paths=[
                map_path,
            ],
            no_pov_effect=True,
            screen_encoding_mode=ScreenEncodingMode.RAW,
            use_vglrun=check_vglrun(),
        ),
        [],
    )
    env = FastResetWrapper(
        TimeLimitWrapper(
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
            max_timesteps=20000,
        ),
    )
    env = DummyVecEnv([lambda: env])
    env = Monitor(env)
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 20000 == 0,
        video_length=20000,
    )

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        device=get_device(),
        tensorboard_log=f"runs/{run.id}",
        gae_lambda=0.99,
        ent_coef=0.005,
        n_steps=512,
    )

    try:
        model.learn(
            total_timesteps=6000000,
            callback=[
                WandbCallback(
                    gradient_save_freq=500,
                    model_save_path=f"models/{run.id}",
                    verbose=2,
                ),
            ],
        )
        model.save(f"{group_name}.ckpt")
        run.finish()
    finally:
        base_env.terminate()


if __name__ == "__main__":
    example_exploration()
```

The `get_device` method is a utility method that returns the device to use for training. It is defined as follows:

```python
import torch

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device
```

The `check_vglrun` method is a utility method that returns whether `vglrun` is installed on the system. It is defined as follows:

```python
def check_vglrun():
    from shutil import which

    return which("vglrun") is not None
```
