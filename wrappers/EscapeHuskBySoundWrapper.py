from typing import SupportsFloat, Any, List

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType

import mydojo
from models.dqn import DQNSoundAgent
from mydojo.minecraft import int_to_action
from wrapper_runner import WrapperRunner


class EscapeHuskBySoundWrapper(gym.Wrapper):
    def __init__(self):
        self.env = mydojo.make(
            initialInventoryCommands=[],
            initialPosition=None,  # nullable
            initialMobsCommands=[
                # "minecraft:sheep",
                "minecraft:husk ~ ~ ~5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
                "minecraft:husk ~5 ~ ~5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
                "minecraft:husk ~-5 ~ ~-5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
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
        super(EscapeHuskBySoundWrapper, self).__init__(self.env)
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(8,), dtype=np.float32
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
        sound_vector = self.encode_sound(sound_subtitles, obs.x, obs.y, obs.z)

        reward = 1  # initial reward
        if is_dead:  #
            if self.initial_env.isHardCore:
                reward = -10000000
                terminated = True
            else:  # send respawn packet
                # pass
                # reward = -200
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

    @staticmethod
    def encode_sound(sound_subtitles, x, y, z) -> List[int]:
        sound_vector = [0] * 8
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
        return sound_vector

    def reset(self, fast_reset: bool = True) -> WrapperObsType:
        obs = self.env.reset(fast_reset=fast_reset)
        obs = obs["obs"]
        sound_subtitles = obs.sound_subtitles
        sound_vector = self.encode_sound(sound_subtitles, obs.x, obs.y, obs.z)
        return np.array(sound_vector, dtype=np.float32)


def main():
    env = EscapeHuskBySoundWrapper()
    buffer_size = 1000000
    batch_size = 256
    gamma = 0.99
    learning_rate = 0.0001
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
        "EscapeHuskSound-6Actions",
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
