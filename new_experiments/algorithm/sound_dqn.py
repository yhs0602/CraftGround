import time
from collections import deque
from typing import Optional

import numpy as np
import torch
from torch import nn, optim

from models.dqn import SoundDQN
from models.replay_buffer import ReplayBuffer
from new_experiments import criterion
from new_experiments.algorithm.epsilon_greedy import EpsilonGreedyExplorer
from new_experiments.logger import Logger


class SoundDQNAlgorithm:
    def __init__(
        self,
        env,
        logger: Logger,
        num_episodes,
        steps_per_episode,
        test_frequency,
        solved_criterion,
        hidden_dim,
        device,
        epsilon_init,
        epsilon_decay,
        epsilon_min,
        update_frequency,
        train_frequency,
        replay_buffer_size,
        batch_size,
        gamma,
        learning_rate,
        weight_decay,
        tau,
    ):
        self.logger = logger
        self.env = env
        self.num_episodes = num_episodes
        self.test_frequency = test_frequency
        self.steps_per_episode = steps_per_episode
        self.state_dim = (np.prod(env.observation_space.shape),)
        self.action_dim = env.action_space.n
        self.hidden_dim = hidden_dim
        self.epsilon_init = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.update_frequency = update_frequency
        self.train_frequency = train_frequency

        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        self.gamma = gamma
        self.tau = tau

        self.batch_size = batch_size

        self.device = device
        self.policy_net = SoundDQN(self.state_dim, self.action_dim, hidden_dim).to(
            device
        )
        self.target_net = SoundDQN(self.state_dim, self.action_dim, hidden_dim).to(
            device
        )
        self.explorer = EpsilonGreedyExplorer(
            self.epsilon_init, self.epsilon_decay, self.epsilon_min
        )

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.loss_fn = nn.MSELoss()

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
                test_score, num_steps, time_took = self.test_agent(episode)
                recent_test_scores.append(test_score)
                avg_test_score = np.mean(recent_test_scores)
                avg_test_scores.append(avg_test_score)
            else:  # training
                (
                    episode_reward,
                    num_steps,
                    time_took,
                    avg_loss,
                    reset_extra_info,
                ) = self.train_agent()
                scores.append(episode_reward)
                recent_scores.append(episode_reward)
                avg_score = np.mean(recent_scores)
                avg_scores.append(avg_score)
                self.logger.log(
                    episode=episode,
                    score=episode_reward,
                    avg_score=avg_score,
                    avg_loss=avg_loss,
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

    def test_agent(self, episode):
        self.logger.before_episode(self.env, should_record_video=True, episode=episode)
        state, reset_info = self.env.reset(fast_reset=True)
        episode_reward = 0
        steps_in_episode = 0
        start_time = time.time()
        for step in range(self.steps_per_episode):
            self.logger.before_step(step, should_record_video=True)
            action = self.exploit_action(state)
            next_state, reward, done, truncated, info = self.env.step(action)
            episode_reward += reward
            steps_in_episode += 1
            if done:
                break
            state = next_state
        time_took = time.time() - start_time
        self.logger.after_episode()
        return episode_reward, steps_in_episode, time_took

    # Train the agent for the given episode using epsilon greedy
    # returns: episode reward, steps, time, loss, reset_extra_info
    def train_agent(self):
        self.logger.before_episode(
            self.env, should_record_video=False, episode=self.episode
        )
        state, reset_info = self.env.reset(fast_reset=True)
        episode_reward = 0
        steps_in_episode = 0
        losses = []
        start_time = time.time()

        for step in range(self.steps_per_episode):
            self.logger.before_step(step, should_record_video=False)
            if self.explorer.should_explore():
                action = np.random.choice(self.action_dim)
            else:  # exploit
                action = self.exploit_action(state)
            next_state, reward, done, truncated, info = self.env.step(action)
            episode_reward += reward
            steps_in_episode += 1
            self.total_steps += 1

            # add experience to replay buffer
            self.replay_buffer.add(state, action, next_state, reward, done)

            # update policy network
            if self.total_steps % self.train_frequency == 0:
                loss = self.update_policy_net()
                losses.append(loss)

            # update target network
            if self.total_steps % self.update_frequency == 0:
                self.update_target_net()

            if done:
                break

        end_time = time.time()
        self.explorer.after_episode()  # update epsilon
        avg_loss = np.mean([loss for loss in losses if loss is not None])
        return (
            episode_reward,
            steps_in_episode,
            end_time - start_time,
            avg_loss,
            reset_info,
        )

    def exploit_action(self, state) -> int:
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state)
        self.policy_net.train()
        return q_values.argmax().item()

    # update the policy network using a batch of experiences
    # returns the loss for logging
    def update_policy_net(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return
        # print("Will update model")
        state, action, next_state, reward, done = self.replay_buffer.sample(
            self.batch_size
        )
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device).squeeze(1)
        next_state = next_state.to(self.device)
        done = done.to(self.device).squeeze(1)

        q_values = self.policy_net(state).gather(1, action.to(torch.int64)).squeeze(1)
        next_q_values = self.target_net(next_state).max(1)[0]
        expected_q_values = reward + (1 - done) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, expected_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target_net(self):
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )
