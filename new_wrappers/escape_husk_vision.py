from typing import SupportsFloat, Any

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType

import mydojo
from mydojo.minecraft import int_to_action
from wrapper_runners.dqn_wrapper_runner import DQNWrapperRunner


class EscapeHusksVisionWrapper(gym.Wrapper):
    def __init__(self, verbose=False, env_path=None):
        self.env = mydojo.make(
            verbose=verbose,
            env_path=env_path,
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
        )
        super().__init__(self.env)
        self.action_space = gym.spaces.Discrete(6)
        initial_env = self.env.initial_env
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, initial_env.imageSizeX, initial_env.imageSizeY),
            dtype=np.uint8,
        )

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action_arr = int_to_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action_arr)
        rgb = obs["rgb"]
        obs = obs["obs"]
        is_dead = obs.is_dead

        reward = 0.5  # initial reward
        if is_dead:  #
            if self.initial_env.isHardCore:
                reward = -100
                terminated = True
            else:  # send respawn packet
                reward = -1
                terminated = True
        return (
            rgb,
            reward,
            terminated,
            truncated,
            info,
        )  # , done: deprecated

    def reset(self, fast_reset: bool = True) -> WrapperObsType:
        obs = self.env.reset(fast_reset=fast_reset)
        return obs["rgb"]


def main():
    env = EscapeHusksVisionWrapper(verbose=False)
    buffer_size = 1000000
    batch_size = 256
    gamma = 0.99
    learning_rate = 0.0001  # 0.001은 너무 크다
    update_freq = 2000  # 에피소드 여러 개 하면서 학습하게 1000 이렇게 하고 줄이기
    hidden_dim = 128  # 128정도 해보기
    weight_decay = 1e-6
    kernel_size = 9
    stride = 2
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    from models.dqn import DQNAgent

    agent = DQNAgent(
        state_dim,
        action_dim,
        hidden_dim,
        kernel_size,
        stride,
        buffer_size,
        batch_size,
        gamma,
        learning_rate,
        weight_decay,
    )
    runner = DQNWrapperRunner(
        env,
        env_name="husk-vision-decay",
        agent=agent,
        max_steps_per_episode=400,
        num_episodes=4000,
        test_frequency=20,
        solved_criterion=lambda avg_score, test_score, avg_test_score, episode: avg_score
        >= 195.0
        and avg_test_score >= 195.0
        and episode >= 1000
        and test_score == 200.0
        if avg_score is not None
        else False and episode >= 1000,
        after_wandb_init=lambda *args: None,
        warmup_episodes=0,
        update_frequency=update_freq,
        epsilon_init=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99,
        resume=False,
        max_saved_models=2,
    )
    runner.run_wrapper(record_video=True)


if __name__ == "__main__":
    main()
