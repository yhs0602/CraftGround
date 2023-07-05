import numpy as np

from final_experiments.get_device import get_device
from final_experiments.wrapper_runners.drqn_wrapper_runner import DRQNWrapperRunner


def train_vision_drqn(
    group,
    agent_class,
    env,
    batch_size,
    time_step,
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
    warmup_episodes,
    solved_criterion,
    stack_size=None,
    **extra_configs,
):
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    agent_instance = agent_class(
        state_dim,
        action_dim,
        hidden_dim,
        kernel_size,
        stride,
        buffer_size,
        batch_size,
        time_step,
        gamma,
        learning_rate,
        weight_decay,
        stack_size=stack_size,
        device=get_device(),
    )

    print("Running DRQN vision wrapper runner")

    runner = DRQNWrapperRunner(
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
