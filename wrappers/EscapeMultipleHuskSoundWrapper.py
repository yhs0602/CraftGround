import gymnasium as gym
import numpy as np

import mydojo
from models.dqn import DQNSoundAgent
from wrapper_runner import WrapperRunner
from wrappers.EscapeHuskBySoundWithPlayerWrapper import (
    EscapeHuskBySoundWithPlayerWrapper,
)


class EscapeMultipleHuskSoundWrapper(EscapeHuskBySoundWithPlayerWrapper):
    def __init__(self):
        super(EscapeMultipleHuskSoundWrapper, self).__init__()
        self.env = mydojo.make(
            initialInventoryCommands=[],
            initialPosition=None,  # nullable
            initialMobsCommands=[
                # "minecraft:sheep",
                "minecraft:husk ~ ~ ~5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
                "minecraft:husk ~5 ~ ~5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
                "minecraft:husk ~-5 ~ ~5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
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

        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(6,), dtype=np.float32
        )


def main():
    env = EscapeMultipleHuskSoundWrapper()
    buffer_size = 1000000
    batch_size = 256
    gamma = 0.99
    learning_rate = 0.0001  # 0.001은 너무 크다
    update_freq = 1000  # 에피소드 여러 개 하면서 학습하게 1000 이렇게 하고 줄이기
    hidden_dim = 128  # 128정도 해보기
    # weight_decay = 0.0001
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    agent = DQNSoundAgent(
        state_dim,
        action_dim,
        hidden_dim,
        buffer_size,
        batch_size,
        gamma,
        learning_rate,
        # weight_decay=weight_decay,
    )
    runner = WrapperRunner(
        env,
        env_name="MultipleHuskSound-6-6-yaw",
        agent=agent,
        max_steps_per_episode=400,
        num_episodes=1500,
        warmup_episodes=0,
        epsilon_init=1,
        epsilon_min=0.01,
        epsilon_decay=0.99,
        update_frequency=update_freq,
        test_frequency=10,
        solved_criterion=lambda avg_score, episode: avg_score >= 190.0
        and episode >= 1000,
    )
    runner.run_wrapper(record_video=True)


if __name__ == "__main__":
    main()
