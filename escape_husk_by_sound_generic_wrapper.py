from wrapper_runners.dqn_wrapper_runner import DQNWrapperRunner


def main():
    from wrappers.EscapeHuskBySoundWithPlayerWrapper import (
        EscapeHuskBySoundWithPlayerWrapper,
    )

    env = EscapeHuskBySoundWithPlayerWrapper()
    buffer_size = 1000000
    batch_size = 256
    gamma = 0.99
    learning_rate = 0.00003  # 0.001은 너무 크다
    update_freq = 2000  # 에피소드 여러 개 하면서 학습하게 1000 이렇게 하고 줄이기
    hidden_dim = 128  # 128정도 해보기
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
        # weight_decay=weight_decay,
    )
    runner = DQNWrapperRunner(
        env,
        env_name="HuskSound-6-6-yaw-fixed",
        agent=agent,
        max_steps_per_episode=400,
        num_episodes=2000,
        test_frequency=20,
        solved_criterion=lambda avg_score, test_score, episode: avg_score >= 195.0
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
