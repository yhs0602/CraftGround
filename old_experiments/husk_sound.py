from old_experiments.train import train_sound

if __name__ == "__main__":
    train_sound(
        verbose=False,
        env_path=None,
        port=8004,
        agent="DQNAgent",
        env_name="husk",
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
        warmup_episodes=0,
    )
