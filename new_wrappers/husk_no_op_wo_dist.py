import math
from typing import List, SupportsFloat, Any

import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperActType, WrapperObsType

import mydojo
from mydojo.minecraft import no_op
from wrapper_runners.dqn_wrapper_runner import DQNWrapperRunner


class HuskSoundNoOpWrapper(gym.Wrapper):
    def __init__(self, verbose=False, env_path=None):
        self.env = mydojo.make(
            verbose=verbose,
            env_path=env_path,
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
            obs_keys=["sound_subtitles"],
        )
        super().__init__(self.env)
        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(7,), dtype=np.float32
        )

    def int_to_action(self, input_act):
        act = no_op()
        # act=0: no op
        if input_act == 1:  # go forward
            act[0] = 1  # 0: noop 1: forward 2 : back
        elif input_act == 2:  # go backward
            act[0] = 2  # 0: noop 1: forward 2 : back
        elif input_act == 3:  # move right
            act[1] = 1  # 0: noop 1: move right 2: move left
        elif input_act == 4:  # move left
            act[1] = 2  # 0: noop 1: move right 2: move left
        elif input_act == 5:  # Turn left
            act[4] = 12 - 1  # Camera delta yaw (0: -180, 24: 180)
        elif input_act == 6:  # Turn right
            act[0] = 12 + 1  # Camera delta yaw (0: -180, 24: 180)
        return act

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action_arr = self.int_to_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action_arr)
        # rgb = obs["rgb"]
        obs = obs["obs"]
        is_dead = obs.is_dead
        sound_subtitles = obs.sound_subtitles
        sound_vector = self.encode_sound_and_yaw(sound_subtitles, obs.x, obs.y, obs.yaw)

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

    @staticmethod
    def encode_sound_and_yaw(
        sound_subtitles, x: float, y: float, yaw: float
    ) -> List[float]:
        sound_vector = [0.0] * 7
        for sound in sound_subtitles:
            if sound.x - x > 16 or sound.y - y > 16:
                continue
            if sound.x - x < -16 or sound.y - y < -16:
                continue
            if sound.translate_key == "subtitles.entity.husk.ambient":
                # normalize
                dx = sound.x - x
                dy = sound.y - y
                distance = math.sqrt(dx * dx + dy * dy)
                if distance > 0:
                    sound_vector[0] = dx / distance
                    sound_vector[1] = dy / distance
            elif sound.translate_key == "subtitles.block.generic.footsteps":
                # normalize
                dx = sound.x - x
                dy = sound.y - y
                distance = math.sqrt(dx * dx + dy * dy)
                if distance > 0:
                    sound_vector[2] = dx / distance
                    sound_vector[3] = dy / distance
            elif sound.translate_key == "subtitles.entity.player.hurt":
                sound_vector[4] = 1
        # Trigonometric encoding
        yaw_radians = math.radians(yaw)
        sound_vector[5] = math.sin(yaw_radians)
        sound_vector[6] = math.cos(yaw_radians)
        return sound_vector

    def reset(self, fast_reset: bool = True) -> WrapperObsType:
        obs = self.env.reset(fast_reset=fast_reset)
        obs = obs["obs"]
        sound_subtitles = obs.sound_subtitles
        sound_vector = self.encode_sound_and_yaw(sound_subtitles, obs.x, obs.y, obs.yaw)
        return np.array(sound_vector, dtype=np.float32)


def main():
    env = HuskSoundNoOpWrapper(verbose=False)
    buffer_size = 1000000
    batch_size = 256
    gamma = 0.99
    learning_rate = 0.00001  # 0.001은 너무 크다
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
        env_name="husk_no_op_wo_dist",
        agent=agent,
        max_steps_per_episode=400,
        num_episodes=2000,
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
