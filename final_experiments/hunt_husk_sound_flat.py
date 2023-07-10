import time

import numpy as np

from env_wrappers.husk_environment import env_makers
from final_experiments.runners.sound import train_sound
from final_experiments.wrappers.action import ActionWrapper
from final_experiments.wrappers.attack_kill import AttackKillWrapper
from final_experiments.wrappers.avoid_damage import AvoidDamageWrapper
from final_experiments.wrappers.sound import SoundWrapper
from models.dueling_sound_dqn import DuelingSoundDQNAgent


def solved_criterion(avg_score, test_score, avg_test_score, episode):
    if episode < 500:
        return False
    if avg_score < 0.95:
        return False
    if test_score < 0.95:
        return False
    if avg_test_score is None:
        return True
    if avg_test_score < 0.95:
        return False
    return True


def run_experiment():
    seed = int(time.time())
    np.random.seed(seed)

    verbose = False
    env_path = None
    port = 8004
    inner_env, sound_list = env_makers["husk-hunt"](verbose, env_path, port, hud=False)
    env = AttackKillWrapper(
        AvoidDamageWrapper(
            SoundWrapper(
                ActionWrapper(
                    inner_env,
                    enabled_actions=[
                        ActionWrapper.NO_OP,
                        ActionWrapper.FORWARD,
                        ActionWrapper.BACKWARD,
                        ActionWrapper.STRAFE_RIGHT,
                        ActionWrapper.STRAFE_LEFT,
                        ActionWrapper.TURN_LEFT,
                        ActionWrapper.TURN_RIGHT,
                        ActionWrapper.JUMP,
                        ActionWrapper.LOOK_UP,
                        ActionWrapper.LOOK_DOWN,
                        ActionWrapper.ATTACK,
                        ActionWrapper.USE,  # use shield
                    ],
                ),
                sound_list=sound_list,
                coord_dim=2,
            )
        )
    )

    train_sound(
        group="hunt_husk_sound_flat",
        env=env,
        agent_class=DuelingSoundDQNAgent,
        # env_name="husk-random-terrain",
        batch_size=256,
        gamma=0.99,
        learning_rate=0.00001,
        update_freq=1000,
        hidden_dim=128,
        weight_decay=0.00001,
        buffer_size=1000000,
        epsilon_init=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.01,
        max_steps_per_episode=400,
        num_episodes=2000,
        warmup_episodes=10,
        solved_criterion=solved_criterion,
    )


if __name__ == "__main__":
    run_experiment()
