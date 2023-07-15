import time
from collections import deque

import numpy as np
import torch

import criterion
from logger import Logger


class A2CAlgorithm:
    actor_optim: torch.optim.Optimizer
    critic_optim: torch.optim.Optimizer
    actor: torch.nn.Module
    critic: torch.nn.Module

    def __init__(
        self,
        env,
        logger: Logger,
        num_episodes: int,
        steps_per_episode: int,
        test_frequency,
        solved_criterion,
        hidden_dim,
        device,
        update_frequency,
        train_frequency,
        gamma,
        **kwargs,
    ):
        self.logger = logger
        self.env = env
        self.num_episodes = num_episodes
        self.test_frequency = test_frequency
        self.steps_per_episode = steps_per_episode
        self.action_dim = env.action_space.n
        self.hidden_dim = hidden_dim

        self.update_frequency = update_frequency
        self.train_frequency = train_frequency

        self.gamma = gamma

        self.device = device

        self.total_steps = 0
        self.episode = 0

        solved_criterion_config = solved_criterion
        criterion_cls = getattr(criterion, solved_criterion_config["name"])
        self.solved_criterion = criterion_cls(**solved_criterion_config["params"])

    def run(self):
        self.logger.start_training()
        recent_scores = deque(maxlen=30)
        recent_test_scores = deque(maxlen=10)
        scores = []
        avg_scores = []
        avg_test_scores = []
        avg_score = None
        avg_test_score = None
        test_score = None
        self.total_steps = 0
        for episode in range(0, self.num_episodes):
            self.episode = episode
            if episode % self.test_frequency == 0:  # testing
                test_score, num_steps, time_took = self.test_agent()
                recent_test_scores.append(test_score)
                avg_test_score = np.mean(recent_test_scores)
                avg_test_scores.append(avg_test_score)
                self.logger.log(
                    {
                        "test/step": episode,
                        "test/score": test_score,
                    }
                )
            else:  # training
                (
                    episode_reward,
                    num_steps,
                    time_took,
                    avg_actor_loss,
                    avg_critic_loss,
                    reset_extra_info,
                ) = self.train_agent()
                scores.append(episode_reward)
                recent_scores.append(episode_reward)
                avg_score = np.mean(recent_scores)
                avg_scores.append(avg_score)
                self.logger.log(
                    {
                        "episode": episode,
                        "score": episode_reward,
                        "avg_score": avg_score,
                        "avg_actor_loss": avg_actor_loss,
                        "avg_critic_loss": avg_critic_loss,
                    }
                )
            if num_steps == 0:
                num_steps = 1
            print(
                f"Seconds per episode{episode}: {time_took}/{num_steps}={time_took / num_steps:.5f} seconds"
            )

            if self.solved_criterion.criterion(
                avg_score, test_score, avg_test_score, episode
            ):
                print(f"Solved in {episode} episodes!")
                break

    def test_agent(self):
        self.logger.before_episode(
            self.env, should_record_video=True, episode=self.episode
        )
        state, reset_info = self.env.reset(fast_reset=True)
        episode_reward = 0
        steps_in_episode = 0
        start_time = time.time()
        for step in range(self.steps_per_episode):
            self.logger.before_step(step, should_record_video=True)
            _, action = self.exploit_action(state)
            next_state, reward, done, truncated, info = self.env.step(action.item())
            episode_reward += reward
            steps_in_episode += 1
            if done:
                break
            state = next_state
        time_took = time.time() - start_time
        self.logger.after_episode()
        return episode_reward, steps_in_episode, time_took

    def train_agent(self):
        self.logger.before_episode(
            self.env, should_record_video=False, episode=self.episode
        )
        state, reset_info = self.env.reset(fast_reset=True)
        episode_reward = 0
        steps_in_episode = 0
        actor_losses = []
        critic_losses = []
        start_time = time.time()

        for step in range(self.steps_per_episode):
            self.logger.before_step(step, should_record_video=False)
            dist, action = self.exploit_action(state)
            next_state, reward, done, truncated, info = self.env.step(action.item())

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = (
                torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            )

            advantage = (
                reward
                + (1 - done) * self.gamma * self.critic(next_state_tensor)
                - self.critic(state_tensor)
            )

            critic_loss = advantage.pow(2).mean()
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            actor_loss = -dist.log_prob(action) * advantage.detach()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            episode_reward += reward
            steps_in_episode += 1

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

            if done:
                break

            state = next_state
        time_took = time.time() - start_time

        avg_actor_loss = np.mean(actor_losses)
        avg_critic_loss = np.mean(critic_losses)
        return (
            episode_reward,
            steps_in_episode,
            time_took,
            avg_actor_loss,
            avg_critic_loss,
            reset_info,
        )

    def exploit_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        return dist, action
