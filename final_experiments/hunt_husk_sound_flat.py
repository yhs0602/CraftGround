from env_wrappers.husk_environment import env_makers
from final_experiments.runners.sound import train_sound
from final_experiments.wrappers.attack_kill import AttackKillWrapper
from final_experiments.wrappers.avoid_damage import AvoidDamageWrapper
from final_experiments.wrappers.simple_navigation import SimpleNavigationWrapper
from final_experiments.wrappers.sound import SoundWrapper
from models.dueling_sound_dqn import DuelingSoundDQNAgent

if __name__ == "__main__":
    verbose = False
    env_path = None
    port = 8004
    inner_env, sound_list = env_makers["husk-hunt"](verbose, env_path, port)
    env = AttackKillWrapper(
        AvoidDamageWrapper(
            SoundWrapper(
                SimpleNavigationWrapper(
                    inner_env, num_actions=SimpleNavigationWrapper.ATTACK + 1
                ),
                sound_list=sound_list,
                coord_dim=2,
            )
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
        max_steps_per_episode=400,
        num_episodes=2000,
        warmup_episodes=10,
    )
