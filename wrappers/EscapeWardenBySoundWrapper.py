from typing import SupportsFloat, Any, List

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType

import mydojo
from models.dqn import DQNSoundAgent
from mydojo.minecraft import int_to_action
from wrapper_runner import WrapperRunner


class EscapeWardenBySoundWrapper(gym.Wrapper):
    def __init__(self):
        self.env = mydojo.make(
            initialInventoryCommands=[],
            initialPosition=None,  # nullable
            initialMobsCommands=[
                # "minecraft:sheep",
                "minecraft:warden ~ ~ ~5",
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
        super(EscapeWardenBySoundWrapper, self).__init__(self.env)
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(52,), dtype=np.float32
        )

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action_arr = int_to_action(action)
        action_arr[2] = 2  # must crawl
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
                reward = -200
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
        sound_vector = [0] * 52
        for sound in sound_subtitles:
            if sound.x - x > 16 or sound.y - y > 16 or sound.z - z > 16:
                continue
            if sound.x - x < -16 or sound.y - y < -16 or sound.z - z < -16:
                continue
            if sound.translate_key == "subtitles.entity.warden.ambient":
                sound_vector[0] = 1
                sound_vector[1] = (sound.x - x) / 16
                sound_vector[2] = (sound.y - y) / 16
                sound_vector[3] = (sound.z - z) / 16
            elif sound.translate_key == "subtitles.entity.warden.agitated":
                sound_vector[4] = 1
                sound_vector[5] = (sound.x - x) / 16
                sound_vector[6] = (sound.y - y) / 16
                sound_vector[7] = (sound.z - z) / 16
            elif sound.translate_key == "subtitles.entity.warden.angry":
                sound_vector[8] = 1
                sound_vector[9] = (sound.x - x) / 16
                sound_vector[10] = (sound.y - y) / 16
                sound_vector[11] = (sound.z - z) / 16
            elif sound.translate_key == "subtitles.entity.warden.listening_angry":
                sound_vector[12] = 1
                sound_vector[13] = (sound.x - x) / 16
                sound_vector[14] = (sound.y - y) / 16
                sound_vector[15] = (sound.z - z) / 16
            elif sound.translate_key == "subtitles.entity.warden.attack_impact":
                sound_vector[16] = 1
                sound_vector[17] = (sound.x - x) / 16
                sound_vector[18] = (sound.y - y) / 16
                sound_vector[19] = (sound.z - z) / 16
            elif sound.translate_key == "subtitles.entity.warden.heartbeat":
                sound_vector[20] = 1
                sound_vector[21] = (sound.x - x) / 16
                sound_vector[22] = (sound.y - y) / 16
                sound_vector[23] = (sound.z - z) / 16
            elif sound.translate_key == "subtitles.entity.warden.listening":
                sound_vector[24] = 1
                sound_vector[25] = (sound.x - x) / 16
                sound_vector[26] = (sound.y - y) / 16
                sound_vector[27] = (sound.z - z) / 16
            elif sound.translate_key == "subtitles.entity.warden.tendril_clicks":
                sound_vector[28] = 1
                sound_vector[29] = (sound.x - x) / 16
                sound_vector[30] = (sound.y - y) / 16
                sound_vector[31] = (sound.z - z) / 16
            elif sound.translate_key == "subtitles.entity.warden.roar":
                sound_vector[32] = 1
                sound_vector[33] = (sound.x - x) / 16
                sound_vector[34] = (sound.y - y) / 16
                sound_vector[35] = (sound.z - z) / 16
            elif sound.translate_key == "subtitles.entity.warden.sniff":
                sound_vector[36] = 1
                sound_vector[37] = (sound.x - x) / 16
                sound_vector[38] = (sound.y - y) / 16
                sound_vector[39] = (sound.z - z) / 16
            elif sound.translate_key == "subtitles.entity.warden.sonic_boom":
                sound_vector[40] = 1
                sound_vector[41] = (sound.x - x) / 16
                sound_vector[42] = (sound.y - y) / 16
                sound_vector[43] = (sound.z - z) / 16
            elif sound.translate_key == "subtitles.entity.warden.sonic_charge":
                sound_vector[44] = 1
                sound_vector[45] = (sound.x - x) / 16
                sound_vector[46] = (sound.y - y) / 16
                sound_vector[47] = (sound.z - z) / 16
            elif sound.translate_key == "subtitles.entity.warden.step":
                sound_vector[48] = 1
                sound_vector[49] = (sound.x - x) / 16
                sound_vector[50] = (sound.y - y) / 16
                sound_vector[51] = (sound.z - z) / 16
        return sound_vector

    def reset(self, fast_reset: bool = True) -> WrapperObsType:
        obs = self.env.reset(fast_reset=fast_reset)
        return np.zeros((52,), dtype=np.float32)


def main():
    env = EscapeWardenBySoundWrapper()
    buffer_size = 1000000
    batch_size = 256
    gamma = 0.99
    learning_rate = 0.0005
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
        "EscapeWardenSound-6Actions",
        agent=agent,
        max_steps_per_episode=1000,
        update_frequency=update_freq,
        solved_criterion=lambda avg_score, episode: avg_score >= 950.0
        and episode >= 100,
    )
    runner.run_wrapper()


if __name__ == "__main__":
    main()
