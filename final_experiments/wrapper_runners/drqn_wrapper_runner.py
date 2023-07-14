import time
from typing import Optional, Union

import numpy as np
import wandb
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

from final_experiments.wrapper_runners.generic_wrapper_runner import (
    GenericWrapperRunner,
)
from models.dueling_bimodal_drqn import DuelingBimodalDRQNAgent
from models.dueling_sound_drqn import DuelingSoundDRQNAgent
from models.transition import Transition
from mydojo.MyEnv import print_with_time


class DRQNWrapperRunner(GenericWrapperRunner):
    agent: Union[DuelingSoundDRQNAgent, DuelingBimodalDRQNAgent]

    def __init__(
        self,
        env,
        env_name,
        group,
        agent: Union[DuelingSoundDRQNAgent, DuelingBimodalDRQNAgent],
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
        transition_class=Transition,
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
        self.transition_class = transition_class

    def train_agent(self, episode, accum_steps):
        state, reset_info = self.env.reset(fast_reset=True)
        # reset_extra_info = reset_info.get("extra_info", None)
        print_with_time("Finished resetting the environment")
        episode_reward = 0
        sum_time = 0
        num_steps = 0
        losses = []
        hidden_state, cell_state = self.agent.policy_net.init_hidden_states(bsize=1)
        for step in range(self.max_steps_per_episode):
            start_time = time.time()
            action, (hidden_state, cell_state) = self.select_action(
                episode, state, False, hidden_state=hidden_state, cell_state=cell_state
            )
            next_state, reward, terminated, truncated, info = self.env.step(action)
            episode_reward += reward
            accum_steps += 1
            num_steps += 1
            loss = self.after_step(
                step,
                accum_steps,
                state,
                action,
                next_state,
                reward,
                terminated,
                truncated,
                info,
                False,
            )
            losses.append(loss)
            if terminated:
                break

            state = next_state
            elapsed_time = time.time() - start_time
            # print(f"Step {step} took {elapsed_time:.5f} seconds")
            sum_time += elapsed_time
        avg_loss = np.mean([loss for loss in losses if loss is not None])
        return (
            episode_reward,
            num_steps,
            accum_steps,
            sum_time,
            avg_loss,
            None,
        )  # reset_extra_info

    def test_agent(self, episode, record_video):
        if record_video:
            video_recorder = VideoRecorder(self.env, f"video{episode}.mp4")
        state, info = self.env.reset(fast_reset=True)
        print_with_time("Finished resetting the environment")
        hidden_state, cell_state = self.agent.policy_net.init_hidden_states(bsize=1)
        episode_reward = 0
        time_took = 0
        num_steps = 0
        for step in range(self.max_steps_per_episode):
            start_time = time.time()
            if record_video:
                video_recorder.capture_frame()
            action, (hidden_state, cell_state) = self.select_action(
                episode, state, True, hidden_state=hidden_state, cell_state=cell_state
            )
            next_state, reward, terminated, truncated, info = self.env.step(action)
            # extra_info = info.get("extra_info", None)
            episode_reward += reward
            self.after_step(
                step,
                step,
                state,
                action,
                next_state,
                reward,
                terminated,
                truncated,
                info,
                True,
            )

            if terminated:
                break
            state = next_state
            elapsed_time = time.time() - start_time
            # print(f"Step {step} took {elapsed_time:.5f} seconds")
            time_took += elapsed_time
            num_steps += 1
        if record_video:
            video_recorder.close()
        to_log = {"test/score": episode_reward, "test/step": episode}
        # to_log.update(reset_extra_info)
        wandb.log(to_log)
        return episode_reward, num_steps, time_took

    def select_action(self, episode, state, testing, **kwargs):
        if episode < self.warmup_episodes:
            action = self.agent.select_action(state, testing, epsilon=1.0, **kwargs)
        else:
            action = self.agent.select_action(
                state, testing, epsilon=self.epsilon, **kwargs
            )
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
            self.transition_class(state, action, next_state, reward, terminated)
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
