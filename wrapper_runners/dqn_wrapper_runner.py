from wrapper_runners.generic_wrapper_runner import GenericWrapperRunner


class DQNWrapperRunner(GenericWrapperRunner):
    def __init__(
        self,
        env,
        env_name,
        agent: "DQNAgent",
        max_steps_per_episode,
        num_episodes,
        test_frequency,
        solved_criterion,
        after_wandb_init: callable,
        warmup_episodes,
        update_frequency,
        epsilon_init,
        epsilon_min,
        epsilon_decay,
        resume: bool = False,
        max_saved_models=2,
    ):
        import models.dqn

        def after_wandb_init_fn():
            models.dqn.after_wandb_init()
            after_wandb_init()

        super().__init__(
            env,
            env_name,
            agent,
            max_steps_per_episode,
            num_episodes,
            test_frequency,
            solved_criterion,
            after_wandb_init_fn,
            resume,
            max_saved_models,
            warmup_episodes=warmup_episodes,
            update_frequency=update_frequency,
            epsilon_init=epsilon_init,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
        )
        self.epsilon = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.warmup_episodes = warmup_episodes
        self.update_frequency = update_frequency

    def select_action(self, episode, state, testing):
        if episode < self.warmup_episodes:
            action = self.env.action_space.sample()
        else:
            action = self.agent.select_action(state, testing, epsilon=self.epsilon)
        return action

    def after_step(
        self,
        step,
        state,
        action,
        next_state,
        reward,
        terminated,
        truncated,
        info,
        testing,
    ):
        if testing:
            return
        self.agent.add_experience(state, action, next_state, reward, terminated)
        self.agent.update_model()
        if step % self.update_frequency == 0:
            self.agent.update_target_model()  # important! FIXME: step ranges from 0 to max_steps_per_episode;

    def after_episode(self, episode, testing: bool):
        if episode >= self.warmup_episodes and not testing:
            self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

    def get_extra_log_info(self):
        return {"epsilon": self.epsilon}
