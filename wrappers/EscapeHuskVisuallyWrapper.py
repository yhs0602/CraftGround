from typing import SupportsFloat, Any

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType

import mydojo
from models.dqn import DQNSoundAgent, DQNAgent
from mydojo.minecraft import int_to_action
from wrapper_runner import WrapperRunner


class EscapeHuskVisuallyWrapper(gym.Wrapper):
    def __init__(self):
        super().__init__()
        self.env = mydojo.make(
            initialInventoryCommands=[],
            initialPosition=None,  # nullable
            initialMobsCommands=[
                # "minecraft:sheep",
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
            obs_keys=[],
        )
        super(EscapeHuskVisuallyWrapper, self).__init__(self.env)
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
                # pass
                reward = -1  # -1 정도로 바꾸기 / 평소에는 0.5 / 좀비 거리 계산해보고
                # TODO: 맞으면 -0.5
                terminated = True
                # send_respawn(self.json_socket)
                # print("Dead!!!!!")
                # res = self.json_socket.receive_json()  # throw away
        return (
            rgb,
            reward,
            terminated,
            truncated,
            info,
        )  # , done: deprecated

    def reset(self, fast_reset: bool = True) -> WrapperObsType:
        obs = self.env.reset(fast_reset=fast_reset)
        obs = obs["obs"]
        return obs["rgb"]


def main():
    env = EscapeHuskVisuallyWrapper()
    buffer_size = 1000000
    batch_size = 256
    gamma = 0.99
    learning_rate = 0.0001  # 0.001은 너무 크다
    update_freq = 1000  # 에피소드 여러 개 하면서 학습하게 1000 이렇게 하고 줄이기 # effectively 400 currently
    hidden_dim = 128  # 128정도 해보기
    kernel_size = 5
    stride = 2
    # weight_decay = 0.0001
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
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
        # weight_decay=weight_decay,
    )
    runner = WrapperRunner(
        env,
        env_name="HuskVis-6",
        agent=agent,
        max_steps_per_episode=400,
        num_episodes=1000,
        warmup_episodes=0,
        epsilon_init=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99,
        update_frequency=update_freq,
        test_frequency=10,
        solved_criterion=lambda avg_score, episode: avg_score >= 190.0
        and episode >= 300,
    )
    runner.run_wrapper(record_video=True)


if __name__ == "__main__":
    main()
