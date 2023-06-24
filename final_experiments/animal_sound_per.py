import time

import numpy as np

from env_wrappers.husk_environment import env_makers
from final_experiments.runners.sound import train_sound
from final_experiments.wrappers.find_animal import FindAnimalWrapper
from final_experiments.wrappers.simplest_navigation import SimplestNavigationWrapper
from final_experiments.wrappers.surrounding_sound import SurroundingSoundWrapper
from models.per_dueling_sound_dqn import PERDuelingSoundDQNAgent


def solved_criterion(avg_score, test_score, avg_test_score, episode):
    if episode < 500:
        return False
    if avg_score < 0.9:
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
    port = 8005
    inner_env, sound_list = env_makers["find-animal"](verbose, env_path, port)
    env = FindAnimalWrapper(
        SurroundingSoundWrapper(
            SimplestNavigationWrapper(
                inner_env, num_actions=SimplestNavigationWrapper.TURN_RIGHT + 1
            ),
            sound_list=sound_list,
            coord_dim=2,
        ),
        target_translation_key="entity.minecraft.chicken",
        target_number=7,
    )

    train_sound(
        group="animal_sound_per",
        env=env,
        agent_class=PERDuelingSoundDQNAgent,
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
        max_steps_per_episode=600,
        num_episodes=3000,
        warmup_episodes=50,
        seed=seed,
        solved_criterion=solved_criterion,
    )


if __name__ == "__main__":
    run_experiment()
