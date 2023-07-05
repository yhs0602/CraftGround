from generic_wrapper_runner import GenericWrapperRunner


class PPOWrapperRunner(GenericWrapperRunner):
    def __init__(
        self,
        env,
        env_name,
        agent: "PPOAgent",
        max_steps_per_episode,
        num_episodes,
        test_frequency,
        solved_criterion,
        after_wandb_init: callable,
        resume: bool = False,
        max_saved_models=2,
    ):
        import models.dqn

        def after_wandb_init_fn():
            models.ppo.after_wandb_init()
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
        )

    def select_action(self, episode, state, testing):
        if episode < self.warmup_episodes:
            action = self.env.action_space.sample()
        else:
            action = self.agent.select_action(state, testing, epsilon=self.epsilon)
        return action

    def after_step(
        self, step, state, action, next_state, reward, terminated, truncated, info
    ):
        self.agent.add_experience(state, action, next_state, reward, terminated)
        self.agent.update_model()
        if step % self.update_frequency == 0:
            self.agent.update_target_model()  # important! FIXME: step ranges from 0 to max_steps_per_episode;

    def after_episode(self, episode, testing: bool):
        if episode >= self.warmup_episodes:
            self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

    def get_extra_log_info(self):
        return {"epsilon": self.epsilon}
