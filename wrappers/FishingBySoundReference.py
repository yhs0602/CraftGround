from collections import deque
from typing import List, SupportsFloat, Any

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType

import mydojo
from models.dqn import DQNSoundAgent
from mydojo.minecraft import no_op
from wrapper_runner import WrapperRunner


class FishingBySoundReferenceWrapper(gym.Wrapper):
    def __init__(self):
        self.env = mydojo.make(
            initialInventoryCommands=[
                "fishing_rod{Enchantments:[{id:lure,lvl:3},{id:mending,lvl:1},{id:unbreaking,lvl:3}]} 1"
            ],
            initialPosition=None,  # nullable
            initialMobsCommands=[],
            imageSizeX=114,
            imageSizeY=64,
            visibleSizeX=114,
            visibleSizeY=64,
            seed=12345,  # nullable
            allowMobSpawn=False,
            alwaysDay=True,
            alwaysNight=False,
            initialWeather="rain",  # raining reduces time to catch fish
            isHardCore=False,
            isWorldFlat=False,  # superflat world
            obs_keys=["sound_subtitles"],
            miscStatKeys=["fish_caught"],
            initialExtraCommands=["tp @p -25 62 -277 127.2 -6.8"],  # x y z yaw pitch
        )
        super(FishingBySoundReferenceWrapper, self).__init__(self.env)
        self.action_space = gym.spaces.Discrete(2)  # 0: no op, 1: reel in
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(5,), dtype=np.float32
        )
        self.caught_fish = deque(maxlen=2)
        self.caught_fish.append(0)
        self.next_action = 1
        self.is_fishing = False

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action_arr = no_op()
        if action == 1:
            action_arr[5] = 1  # use item
        obs, reward, terminated, truncated, info = self.env.step(action_arr)
        # rgb = obs["rgb"]
        obs = obs["obs"]
        sound_subtitles = obs.sound_subtitles
        sound_vector = self.encode_sound(sound_subtitles)

        reward = 0.001 if self.next_action == action else -0.001  # manual guide
        # TODO: guide to fish quickly
        if sound_vector[2] == 1 or sound_vector[4] == 1:  # splash or item pickup
            self.next_action = 1
        else:
            self.next_action = 0

        # check if fish is caught
        fish_caught = obs.misc_statistics["fish_caught"]
        # print("fish caught:", fish_caught)
        self.caught_fish.append(fish_caught)

        if self.caught_fish[0] < self.caught_fish[1]:
            reward += 1.0
            # terminated = True

        # Durability
        # Reeling in while the bobber is in the air or in the water with nothing caught on it costs no durability.
        # Reeling in and successfully catching fish, junk or treasure costs 1 durability.
        # Reeling in while the bobber is stuck in a solid block costs 2 durability.
        # Reeling in a dropped item costs 3 durability.
        # Reeling in an entity that is not a dropped item costs 5 durability.

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
        sound_vector = self.encode_sound(sound_subtitles)
        self.caught_fish = deque(maxlen=2)
        self.caught_fish.append(0)
        self.next_action = 1
        return np.array(sound_vector, dtype=np.float32)

    @staticmethod
    def encode_sound(sound_subtitles) -> List[float]:
        sound_vector = [0.0] * 5
        for sound in sound_subtitles:
            if (
                sound.translate_key == "subtitles.entity.experience_orb.pickup"
            ):  # experience orb gained
                sound_vector[0] = 1
            elif (
                sound.translate_key == "subtitles.entity.fishing_bobber.retrieve"
            ):  # fishing rod reeled in
                sound_vector[1] = 1
            elif (
                sound.translate_key == "subtitles.entity.fishing_bobber.splash"
            ):  # fish bobber splashed
                sound_vector[2] = 1
            elif (
                sound.translate_key == "subtitles.entity.fishing_bobber.throw"
            ):  # fishing rod thrown
                sound_vector[3] = 1
            elif (
                sound.translate_key == "subtitles.entity.item.pickup"
            ):  # item picked up
                sound_vector[4] = 1
        return sound_vector


def main():
    env = FishingBySoundReferenceWrapper()
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
    num_steps_per_episode = 600
    # 100 ~ 600 tick for fishing, 20% discount for rain, 100 ticks discount per lure level
    # 80 ~ 480
    # 0 ~ 180 tick per fish
    # effectively 0~180 ticks = 9 seconds per fish, 3.6 chances to catch fish

    runner = WrapperRunner(
        env,
        env_name="FishingSound-5-2",
        agent=agent,
        max_steps_per_episode=num_steps_per_episode,
        num_episodes=1000,
        warmup_episodes=0,
        epsilon_init=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99,
        update_frequency=update_freq,
        test_frequency=10,
        solved_criterion=lambda avg_score, episode: avg_score >= 2.5 and episode >= 300,
    )
    runner.run_wrapper(record_video=True)


if __name__ == "__main__":
    main()
