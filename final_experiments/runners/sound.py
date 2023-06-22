import numpy as np

from env_experiments.get_device import get_device


def train_sound(
    group,
    agent_class,
    env,
    batch_size,
    gamma,
    learning_rate,
    update_freq,
    hidden_dim,
    weight_decay,
    buffer_size,
    epsilon_init,
    epsilon_decay,
    epsilon_min,
    max_steps_per_episode,
    num_episodes,
    warmup_episodes,
    solved_criterion,
    stack_size=None,
    **extra_configs,
):
    state_dim = env.observation_space.shape
    state_dim = (np.prod(state_dim),)
    action_dim = env.action_space.n
    agent_instance = agent_class(
        state_dim,
        action_dim,
        hidden_dim,
        buffer_size,
        batch_size,
        gamma,
        learning_rate,
        weight_decay,
        stack_size=stack_size,
        device=get_device(),
    )

    from wrapper_runners.dqn_wrapper_runner import DQNWrapperRunner

    print("Running DQN wrapper runner")
    runner = DQNWrapperRunner(
        env,
        env_name="wrapped",
        group=group,
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
