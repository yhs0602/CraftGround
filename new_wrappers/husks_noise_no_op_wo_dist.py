import math
from typing import List, SupportsFloat, Any

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType

import mydojo
from mydojo.minecraft import int_to_action_with_no_op
from final_experiments.wrapper_runners import DQNWrapperRunner

sound_list = [
    "subtitles.entity.sheep.ambient",  # sheep ambient sound
    "subtitles.block.generic.footsteps",  # player, animal walking
    "subtitles.block.generic.break",  # sheep eating grass
    "subtitles.entity.cow.ambient",  # cow ambient sound
    # "subtitles.entity.pig.ambient",  # pig ambient sound
    # "subtitles.entity.chicken.ambient",  # chicken ambient sound
    # "subtitles.entity.chicken.egg",  # chicken egg sound
    "subtitles.entity.husk.ambient",  # husk ambient sound
]


#     "subtitles.entity.player.hurt"  # player hurt sound


def encode_sound(sound_subtitles: List, x: float, z: float, yaw: float) -> List[float]:
    sound_vector = [0] * (len(sound_list) * 2 + 3)
    for sound in sound_subtitles:
        if sound.x - x > 16 or sound.z - z > 16:
            continue
        if sound.x - x < -16 or sound.z - z < -16:
            continue
        for idx, translation_key in enumerate(sound_list):
            if translation_key == sound.translate_key:
                dx = sound.x - x
                dz = sound.z - z
                distance = math.sqrt(dx * dx + dz * dz)
                if distance > 0:
                    sound_vector[idx * 2] = dx / distance
                    sound_vector[idx * 2 + 1] = dz / distance
            elif translation_key == "subtitles.entity.player.hurt":
                sound_vector[-1] = 1  # player hurt sound

    # Trigonometric encoding
    yaw_radians = math.radians(yaw)
    sound_vector[-3] = math.cos(yaw_radians)
    sound_vector[-2] = math.sin(yaw_radians)

    return sound_vector


class HusksWithNoiseSoundWrapper(gym.Wrapper):
    def __init__(self, verbose=False, env_path=None, port=8000):
        self.env = mydojo.make(
            verbose=verbose,
            env_path=env_path,
            port=port,
            initialInventoryCommands=[],
            initialPosition=None,  # nullable
            initialMobsCommands=[
                "minecraft:sheep ~ ~ 5",
                "minecraft:cow ~ ~ -5",
                "minecraft:cow ~5 ~ -5",
                "minecraft:sheep ~-5 ~ -5",
                "minecraft:husk ~ ~ ~5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
                "minecraft:husk ~5 ~ ~5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
                "minecraft:husk ~-5 ~ ~-5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
                "minecraft:husk ~-5 ~ ~5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
                "minecraft:husk ~-5 ~ ~ {HandItems:[{Count:1,id:iron_shovel},{}]}",
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
        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(len(sound_list) * 2 + 3,), dtype=np.float32
        )

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action_arr = int_to_action_with_no_op(action)
        obs, reward, terminated, truncated, info = self.env.step(action_arr)
        # rgb = obs["rgb"]
        obs = obs["obs"]
        is_dead = obs.is_dead
        sound_subtitles = obs.sound_subtitles
        sound_vector = encode_sound(sound_subtitles, obs.x, obs.z, obs.yaw)

        reward = 0.5  # initial reward
        if is_dead:  #
            if self.initial_env.isHardCore:
                reward = -100
                terminated = True
            else:  # send respawn packet
                reward = -1
                terminated = True
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
        sound_vector = encode_sound(sound_subtitles, obs.x, obs.z, obs.yaw)
        return np.array(sound_vector, dtype=np.float32)


def main():
    env = HusksWithNoiseSoundWrapper(verbose=False, port=8005)
    buffer_size = 1000000
    batch_size = 256
    gamma = 0.99
    learning_rate = 0.000005  # 0.001은 너무 크다
    update_freq = 2400  # 에피소드 여러 개 하면서 학습하게 1000 이렇게 하고 줄이기
    hidden_dim = 128  # 128정도 해보기
    weight_decay = 1e-5
    # weight_decay = 0.0001
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    from models.dqn import DQNSoundAgent

    agent = DQNSoundAgent(
        state_dim,
        action_dim,
        hidden_dim,
        buffer_size,
        batch_size,
        gamma,
        learning_rate,
        weight_decay,
    )
    runner = DQNWrapperRunner(
        env,
        env_name="husk_noise_wo_dist",
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
        epsilon_min=0.01,
        epsilon_decay=0.99,
        resume=False,
        max_saved_models=2,
    )
    runner.run_wrapper(record_video=True)


if __name__ == "__main__":
    main()
