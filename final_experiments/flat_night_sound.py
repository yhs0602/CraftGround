import time

import numpy as np

from env_wrappers.husk_environment import env_makers
from final_experiments.runners.sound import train_sound
from final_experiments.wrappers.avoid_damage import AvoidDamageWrapper
from final_experiments.wrappers.simple_navigation import SimpleNavigationWrapper
from final_experiments.wrappers.sound import SoundWrapper
from models.dueling_sound_dqn import DuelingSoundDQNAgent


def run_experiment():
    seed = int(time.time())
    np.random.seed(seed)

    verbose = False
    env_path = None
    port = 8000
    inner_env, sound_list = env_makers["flat-night"](verbose, env_path, port)
    env = AvoidDamageWrapper(
        SoundWrapper(
            SimpleNavigationWrapper(
                inner_env, num_actions=SimpleNavigationWrapper.TURN_RIGHT + 1
            ),
            sound_list=sound_list,
            coord_dim=2,
        )
    )

    train_sound(
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
        max_steps_per_episode=4000,
        num_episodes=2000,
        warmup_episodes=10,
        group="flat_night_sound",
        seed=seed,
        solved_criterion=lambda avg_score, test_score, avg_test_score, episode: avg_score
        >= 1950.0
        and avg_test_score >= 1950.0
        and episode >= 500
        and test_score >= 1980.0
        if avg_score is not None
        else False and episode >= 500,
    )


if __name__ == "__main__":
    run_experiment()
