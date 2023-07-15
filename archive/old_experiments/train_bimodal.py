from final_experiments.get_device import get_device
from env_wrappers.husk_environment import env_makers
from env_wrappers.vision_and_sound_wrapper import VisionAndSoundWrapper


def train_vision_and_sound(
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
):
    env, sound_list = env_makers[env_name](verbose, env_path, port)
    wrapper = VisionAndSoundWrapper(
        env,
        action_dim=7,
        sound_list=sound_list,
        coord_dim=2,
        reward_function=reward_function,
    )
    if agent == "DQNAgent":
        from models.dqn import MultimodalDQNAgent

        agent_class = MultimodalDQNAgent
    elif agent == "DDQNAgent":
        from models.dqn import MultimodalDQNAgent

        agent_class = MultimodalDQNAgent
    elif agent == "DuelingDQNAgent":
        from models.dueling_bimodal_dqn import DuelingBiModalDQNAgent

        agent_class = DuelingBiModalDQNAgent
    elif agent == "DuelingDQNAttentionAgent":
        from models.dueling_bimodal_attention_dqn import DuelingBiModalAttentionAgent

        agent_class = DuelingBiModalAttentionAgent
    else:
        print(f"Agent not implemented: {agent}")
        raise NotImplementedError
    # print(f"{wrapper.observation_space=}")
    state_dim = wrapper.observation_space["vision"].shape
    sound_dim = wrapper.observation_space["sound"].shape
    action_dim = wrapper.action_space.n
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

    from final_experiments.wrapper_runners.dqn_wrapper_runner import DQNWrapperRunner

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


if __name__ == "__main__":
    train_vision_and_sound(
        verbose=False,
        env_path=None,
        port=8003,
        agent="DQNAgent",
        env_name="husk",
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
        warmup_episodes=0,
    )
