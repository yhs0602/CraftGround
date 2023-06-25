from typing import Optional

from models.dqn import Transition
from models.dueling_sound_drqn import DuelingSoundDRQNAgent
from wrapper_runners.generic_wrapper_runner import GenericWrapperRunner


class DRQNWrapperRunner(GenericWrapperRunner):
    agent: DuelingSoundDRQNAgent

    def __init__(
        self,
        env,
        env_name,
        group,
        agent: DuelingSoundDRQNAgent,
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
        **extra_configs,
    ):
        import models.dqn

        def after_wandb_init_fn():
            models.dqn.after_wandb_init()
            after_wandb_init()

        super().__init__(
            env,
            env_name,
            group,
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
            extra_configs=extra_configs,
        )
        self.epsilon = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.warmup_episodes = warmup_episodes
        self.update_frequency = update_frequency
        self.episode_buffer = []

    def select_action(self, episode, state, testing):
        if episode < self.warmup_episodes:
            action = self.env.action_space.sample()
        else:
            action = self.agent.select_action(state, testing, epsilon=self.epsilon)
        return action

    def after_step(
        self,
        step,
        accum_steps,
        state,
        action,
        next_state,
        reward,
        terminated,
        truncated,
        info,
        testing,
    ) -> Optional[float]:
        if testing:
            return None
        self.episode_buffer.append(
            Transition(state, action, next_state, reward, terminated)
        )
        loss = self.agent.update_model()
        if accum_steps % self.update_frequency == 0:
            print(f"{accum_steps=}, {step=}, {self.update_frequency=}")
            self.agent.update_target_model()
        return loss

    def after_episode(self, episode, testing: bool):
        if testing:
            return
        self.agent.add_episode(self.episode_buffer)
        self.episode_buffer = []
        if episode >= self.warmup_episodes:
            self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

    def get_extra_log_info(self):
        return {"epsilon": self.epsilon}
