from gymnasium.wrappers import FrameStack

from env_wrappers.husk_environment import env_makers
from env_wrappers.vision_wrapper import VisionWrapper


def train_cnn(
    verbose,
    env_path,
    port,
    agent,
    env_name,
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
    warmup_episodes,
    reward_function=None,
    stack_size=None,
):
    env, _ = env_makers[env_name](verbose, env_path, port)
    wrapper = VisionWrapper(env, action_dim=7, reward_function=reward_function)
    if stack_size is not None:
        wrapper = FrameStack(wrapper, stack_size)
    if agent == "DQNAgent":
        from models.dqn import DQNAgent

        agent_class = DQNAgent
    elif agent == "DDQNAgent":
        from models.dqn import DDQNAgent

        agent_class = DDQNAgent
    else:
        print(f"Agent not implemented: {agent}")
        raise NotImplementedError
    state_dim = wrapper.observation_space.shape
    action_dim = wrapper.action_space.n
    agent_instance = agent_class(
        state_dim,
        action_dim,
        hidden_dim,
        kernel_size,
        stride,
        buffer_size,
        batch_size,
        gamma,
        learning_rate,
        weight_decay,
    )

    from wrapper_runners.dqn_wrapper_runner import DQNWrapperRunner

    runner = DQNWrapperRunner(
        wrapper,
        env_name=env_name,
        agent=agent_instance,
        max_steps_per_episode=max_steps_per_episode,
        num_episodes=num_episodes,
        test_frequency=20,
        solved_criterion=lambda avg_score, test_score, avg_test_score, episode: avg_score
        >= 195.0
        and avg_test_score >= 195.0
        and episode >= 1000
        and test_score == 200.0
        if avg_score is not None
        else False and episode >= 1000,
        after_wandb_init=lambda *args: None,
        warmup_episodes=warmup_episodes,
        update_frequency=update_freq,
        epsilon_init=epsilon_init,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        resume=False,
        max_saved_models=1,
    )
    runner.run_wrapper(record_video=True)
