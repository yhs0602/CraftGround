import time

import numpy as np

from env_wrappers.husk_environment import env_makers
from final_experiments.runners.vision import train_cnn
from final_experiments.wrappers.action import ActionWrapper
from final_experiments.wrappers.avoid_damage import AvoidDamageWrapper
from final_experiments.wrappers.find_village import FindVillageWrapper
from final_experiments.wrappers.fly_helper import FlyHelperWrapper
from final_experiments.wrappers.vision import VisionWrapper
from models.dueling_vision_dqn import DuelingVisionDQNAgent
from models.manual_instruction import ManualVisionAgent


def solved_criterion(avg_score, test_score, avg_test_score, episode):
    if episode < 500:
        return False
    if avg_score < 195.0:
        return False
    if test_score < 198.0:
        return False
    if avg_test_score is None:
        return True
    if avg_test_score < 195.0:
        return False
    return True


def run_experiment():
    seed = int(time.time())
    np.random.seed(seed)

    verbose = False
    env_path = None
    port = 8001
    inner_env, sound_list = env_makers["find-village"](
        verbose, env_path, port, hud_hidden=False
    )
    env = FindVillageWrapper(
        FlyHelperWrapper(
            VisionWrapper(
                ActionWrapper(
                    inner_env,
                    enabled_actions=[
                        ActionWrapper.JUMP,
                        ActionWrapper.NO_OP,
                        ActionWrapper.JUMP_USE,
                        ActionWrapper.TURN_RIGHT,
                        ActionWrapper.TURN_LEFT,
                        ActionWrapper.LOOK_UP,
                        ActionWrapper.LOOK_DOWN,
                    ],
                ),
                x_dim=114,
                y_dim=64,
            )
        )
    )

    train_cnn(
        group="elytra_vision",
        env=env,
        agent_class=ManualVisionAgent,
        # env_name="husk-random-terrain",
        batch_size=256,
        gamma=0.99,
        learning_rate=0.00001,
        update_freq=1000,
        hidden_dim=128,
        kernel_size=5,
        stride=2,
        weight_decay=0.00001,
        buffer_size=1000000,
        epsilon_init=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.01,
        max_steps_per_episode=4000,
        num_episodes=1000,
        warmup_episodes=0,
        seed=seed,
        solved_criterion=solved_criterion,
    )


if __name__ == "__main__":
    run_experiment()
