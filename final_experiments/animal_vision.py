from env_wrappers.husk_environment import env_makers
from final_experiments.runners.vision import train_cnn
from final_experiments.wrappers.find_animal import FindAnimalWrapper
from final_experiments.wrappers.simple_navigation import SimpleNavigationWrapper
from final_experiments.wrappers.vision import VisionWrapper
from models.dueling_vision_dqn import DuelingVisionDQNAgent

if __name__ == "__main__":
    verbose = False
    env_path = None
    port = 8001
    inner_env, sound_list = env_makers["find-animal"](verbose, env_path, port)
    env = FindAnimalWrapper(
        VisionWrapper(
            SimpleNavigationWrapper(
                inner_env, num_actions=SimpleNavigationWrapper.JUMP + 1
            ),
            x_dim=114,
            y_dim=64,
        ),
        target_translation_key="entity.minecraft.chicken",
        target_number=7,
    )

    train_cnn(
        env=env,
        agent_class=DuelingVisionDQNAgent,
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
        max_steps_per_episode=400,
        num_episodes=2000,
        warmup_episodes=10,
    )
