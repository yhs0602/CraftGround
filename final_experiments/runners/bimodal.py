from final_experiments.get_device import get_device


def train_vision_and_sound(
    group,
    agent_class,
    env,
    batch_size,
    gamma,
    learning_rate,
    update_freq,
    hidden_dim,
    kernel_size,
    stride,
    weight_decay,
    buffer_size,
    epsilon_init,
    epsilon_decay,
    epsilon_min,
    max_steps_per_episode,
    num_episodes,
    solved_criterion,
    warmup_episodes,
    **extra_configs,
):
    state_dim = env.observation_space["vision"].shape
    sound_dim = env.observation_space["sound"].shape
    action_dim = env.action_space.n
    agent_instance = agent_class(
        state_dim,
        sound_dim,
        action_dim,
        hidden_dim,
        kernel_size,
        stride,
        buffer_size,
        batch_size,
        gamma,
        learning_rate,
        weight_decay,
        device=get_device(),
    )

    from wrapper_runners.dqn_wrapper_runner import DQNWrapperRunner

    runner = DQNWrapperRunner(
        env,
        group=group,
        env_name="wrapped-bimodal",
        agent=agent_instance,
        max_steps_per_episode=max_steps_per_episode,
        num_episodes=num_episodes,
        test_frequency=20,
        solved_criterion=solved_criterion,
        after_wandb_init=lambda *args: None,
        warmup_episodes=warmup_episodes,
        update_frequency=update_freq,
        epsilon_init=epsilon_init,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        resume=False,
        max_saved_models=1,
        extra_configs=extra_configs,
    )
    runner.run_wrapper(record_video=True)
