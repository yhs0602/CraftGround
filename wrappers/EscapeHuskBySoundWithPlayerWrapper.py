from typing import List, SupportsFloat, Any

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType

from models.dqn import DQNSoundAgent
from mydojo.minecraft import int_to_action
from wrapper_runner import WrapperRunner
from wrappers.EscapeHuskBySoundWrapper import EscapeHuskBySoundWrapper


class EscapeHuskBySoundWithPlayerWrapper(EscapeHuskBySoundWrapper):
    def __init__(self, verbose=False, env_path=None, port=8000):
        super().__init__(verbose, env_path, port)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(6,), dtype=np.float32
        )

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action_arr = int_to_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action_arr)
        # rgb = obs["rgb"]
        obs = obs["obs"]
        is_dead = obs.is_dead
        sound_subtitles = obs.sound_subtitles
        sound_vector = self.encode_sound_and_yaw(sound_subtitles, obs.x, obs.z, obs.yaw)

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
            np.array(sound_vector, dtype=np.float32),
            reward,
            terminated,
            truncated,
            info,
        )  # , done: deprecated

    def reset(self, fast_reset: bool = True) -> WrapperObsType:
        obs = self.env.reset(fast_reset=fast_reset)
        obs = obs["obs"]
        sound_subtitles = obs.sound_subtitles
        sound_vector = self.encode_sound_and_yaw(sound_subtitles, obs.x, obs.z, obs.yaw)
        return np.array(sound_vector, dtype=np.float32)

    @staticmethod
    def encode_sound_and_yaw(
        sound_subtitles, x: float, z: float, yaw: float
    ) -> List[float]:
        sound_vector = [0.0] * 6
        for sound in sound_subtitles:
            if sound.x - x > 16 or sound.z - z > 16:
                continue
            if sound.x - x < -16 or sound.z - z < -16:
                continue
            if sound.translate_key == "subtitles.entity.husk.ambient":
                sound_vector[0] = (sound.x - x) / 16
                sound_vector[1] = (sound.z - z) / 16
            elif sound.translate_key == "subtitles.block.generic.footsteps":
                sound_vector[2] = (sound.x - x) / 16
                sound_vector[3] = (sound.z - z) / 16
            elif sound.translate_key == "subtitles.entity.player.hurt":
                sound_vector[4] = 1
        sound_vector[5] = yaw / 180.0
        return sound_vector


def main():
    env = EscapeHuskBySoundWithPlayerWrapper()
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
        env_name="HuskSound-6-6-yaw",
        agent=agent,
        max_steps_per_episode=400,
        num_episodes=700,
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
