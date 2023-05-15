from typing import List

import gymnasium as gym
import numpy as np

from models.dqn import DQNSoundAgent
from wrapper_runner import WrapperRunner
from wrappers.EscapeHuskBySoundWrapper import EscapeHuskBySoundWrapper


class EscapeHuskBySoundWithPlayerWrapper(EscapeHuskBySoundWrapper):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(12,), dtype=np.float32
        )

    @staticmethod
    def encode_sound(sound_subtitles, x, y, z) -> List[int]:
        sound_vector = [0] * 12
        for sound in sound_subtitles:
            if sound.x - x > 16 or sound.y - y > 16 or sound.z - z > 16:
                continue
            if sound.x - x < -16 or sound.y - y < -16 or sound.z - z < -16:
                continue
            if sound.translate_key == "subtitles.entity.husk.ambient":
                sound_vector[0] = 1
                sound_vector[1] = (sound.x - x) / 16
                sound_vector[2] = (sound.y - y) / 16
                sound_vector[3] = (sound.z - z) / 16
            elif sound.translate_key == "subtitles.block.generic.footsteps":
                sound_vector[4] = 1
                sound_vector[5] = (sound.x - x) / 16
                sound_vector[6] = (sound.y - y) / 16
                sound_vector[7] = (sound.z - z) / 16
            elif sound.translate_key == "subtitles.entity.player.hurt":
                sound_vector[8] = 1
                sound_vector[9] = (sound.x - x) / 16
                sound_vector[10] = (sound.y - y) / 16
                sound_vector[11] = (sound.z - z) / 16
        return sound_vector


def main():
    env = EscapeHuskBySoundWithPlayerWrapper()
    buffer_size = 1000000
    batch_size = 256
    gamma = 0.99
    learning_rate = 0.001
    update_freq = 25
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    agent = DQNSoundAgent(
        state_dim,
        action_dim,
        buffer_size,
        batch_size,
        gamma,
        learning_rate,
    )
    runner = WrapperRunner(
        env,
        "EscapeHuskSound-6Actions-12SoundFeatures",
        agent=agent,
        max_steps_per_episode=400,
        num_episodes=3000,
        epsilon_decay=0.999,
        update_frequency=update_freq,
        solved_criterion=lambda avg_score, episode: avg_score >= 380.0
        and episode >= 100,
    )
    runner.run_wrapper(record_video=True)


if __name__ == "__main__":
    main()
