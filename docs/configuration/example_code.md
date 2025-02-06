---
title: Minimal Example
parent: Configuration
---


# Minimal Example

- Please check [the example repository](https://github.com/yhs0602/minecraft-simulator-benchmark) for benchmarking experiments.

```python
import argparse
import sys
import time
from craftground.screen_encoding_modes import ScreenEncodingMode
import wandb
from craftground.wrappers.vision import VisionWrapper

try:
    from craftground.minecraft import no_op_v2
except ImportError:
    from craftground.environment.action_space import no_op_v2
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common import on_policy_algorithm
import platform
from craftground.initial_environment_config import DaylightMode
import gymnasium as gym
from typing import SupportsFloat, Any
from gymnasium.core import WrapperActType, WrapperObsType
import craftground
from craftground import InitialEnvironmentConfig, ActionSpaceVersion
import torch

# Utility function for headless environment setup
def check_vglrun():
    from shutil import which

    return which("vglrun") is not None

# Utility function to get the accelerator device
def get_device(dev_num: int = 0) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{dev_num}")
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

# Utility function to create the CraftGround environment
def make_craftground_env(
    port: int,
    width: int,
    height: int,
    screen_encoding_mode: ScreenEncodingMode,
    verbose_python: bool = False,
    verbose_gradle: bool = False,
    verbose_jvm: bool = False,
) -> gym.Env:
    return craftground.make(
        port=port,
        initial_env_config=InitialEnvironmentConfig(
            image_width=width,
            image_height=height,
            hud_hidden=False,
            render_distance=11,
            screen_encoding_mode=screen_encoding_mode,
        ).set_daylight_cycle_mode(DaylightMode.ALWAYS_DAY),
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
        use_vglrun=check_vglrun(),
        verbose_python=verbose_python,
        verbose_gradle=verbose_gradle,
        verbose_jvm=verbose_jvm,
    )

# A wrapper that converts the dictionary action space to a multidiscrete action space, to be used with stable-baselines3's PPO.
class TreeWrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        super().__init__(env)
        self.action_space = gymnasium.spaces.MultiDiscrete(
            [
                2,  # forward
                2,  # back
                2,  # left
                2,  # right
                2,  # jump
                2,  # sneak
                2,  # sprint
                2,  # attack
                25,  # pitch
                25,  # yaw
            ]
        )

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action_v2 = no_op_v2()
        # print(f"{action=}")
        action_v2["forward"] = action[0]
        action_v2["back"] = action[1]
        action_v2["left"] = action[2]
        action_v2["right"] = action[3]
        action_v2["jump"] = action[4]
        action_v2["sneak"] = action[5]
        action_v2["sprint"] = action[6]
        action_v2["attack"] = action[7]
        action_v2["camera_pitch"] = (action[8] - 12) * 15
        action_v2["camera_yaw"] = (action[9] - 12) * 15
        obs, reward, terminated, truncated, info = self.env.step(action_v2)
        return obs, reward, terminated, truncated, info

# The main function that runs the PPO algorithm with the CraftGround environment
def ppo_check(
    run: wandb.Run,
    screen_encoding_mode: ScreenEncodingMode,
    vision_width: int,
    vision_height: int,
    port: int,
    device: str,
    max_steps: int = 10000,
):
    env = make_craftground_env(
        port=port,
        width=vision_width,
        height=vision_height,
        screen_encoding_mode=screen_encoding_mode,
    )
    # VisionWrapper selects only the visual information from the observation space.
    env = VisionWrapper(env, x_dim=vision_width, y_dim=vision_height)
    # Task specific wrappers
    env = TreeWrapper(env)
    env = DummyVecEnv([lambda: env])
    # Record video every 2000 steps and save the video
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 2000 == 0,
        video_length=2000,
    )
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        device=device,
        tensorboard_log=f"runs/{run.id}",
        gae_lambda=0.99,
        ent_coef=0.005,
        n_steps=512,
    )

    try:
        env.reset()
        model.learn(
            total_timesteps=max_steps,
            callback=[
                # CustomWandbCallback(),
                WandbCallback(
                    gradient_save_freq=500,
                    model_save_path=f"models/{run.id}",
                    verbose=2,
                ),
            ],
        )
        model.save(f"ckpts/{run.group}-{run.name}.ckpt")
    finally:
        env.close()
        run.finish()

if __name__ == "__main__":
    # Initialize wandb
    run = wandb.init(project="craftground", entity="your_entity")
    # Run the PPO algorithm
    ppo_check(
        run=run,
        screen_encoding_mode=ScreenEncodingMode.RAW,
        vision_width=64,
        vision_height=64,
        port=8023,
        device=get_device(),
        max_steps=10000,
    )
```