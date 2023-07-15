import abc
import time
from abc import abstractmethod
from collections import deque
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch import nn

from code.models import RecurrentReplayBuffer
from code.models.transition import Transition, Episode
from code import criterion
from code.algorithm.epsilon_greedy import EpsilonGreedyExplorer
from code.logger import Logger


class DRQNAlgorithm(abc.ABC):
    policy_net: torch.nn.Module
    target_net: torch.nn.Module
    optimizer: torch.optim.Optimizer

    @abstractmethod
    def __init__(
        self,
        env,
        logger: Logger,
        num_episodes: int,
        warmup_episodes: int,
        steps_per_episode: int,
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
        time_step,
        gamma,
        learning_rate,
        weight_decay,
        tau,
        **kwargs,
    ):
        self.logger = logger
        self.env = env
        self.num_episodes = num_episodes
        self.warmup_episodes = warmup_episodes
        self.test_frequency = test_frequency
        self.steps_per_episode = steps_per_episode
        self.action_dim = env.action_space.n
        self.hidden_dim = hidden_dim
        self.epsilon_init = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.update_frequency = update_frequency
        self.train_frequency = train_frequency

        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = RecurrentReplayBuffer(replay_buffer_size)

        self.gamma = gamma
        self.tau = tau

        self.batch_size = batch_size
        self.time_step = time_step

        self.device = device
        self.explorer = EpsilonGreedyExplorer(
            self.epsilon_init, self.epsilon_decay, self.epsilon_min
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
                    avg_loss,
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
                        "avg_loss": avg_loss,
                        "epsilon": self.explorer.epsilon,
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

    def test_agent(self, episode):
        self.logger.before_episode(self.env, should_record_video=True, episode=episode)
        state, reset_info = self.env.reset(fast_reset=True)
        hidden_state, cell_state = self.policy_net.init_hidden_states(bsize=1)
        episode_reward = 0
        steps_in_episode = 0
        start_time = time.time()
        for step in range(self.steps_per_episode):
            self.logger.before_step(step, should_record_video=True)
            action, (hidden_state, cell_state) = self.get_next_hidden_state(
                state, hidden_state, cell_state
            )
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
        hidden_state, cell_state = self.policy_net.init_hidden_states(bsize=1)
        episode_reward = 0
        steps_in_episode = 0
        losses = []
        start_time = time.time()

        episode = []
        for step in range(self.steps_per_episode):
            self.logger.before_step(step, should_record_video=False)
            action, (hidden_state, cell_state) = self.get_next_hidden_state(
                state, hidden_state, cell_state
            )
            if self.explorer.should_explore() or self.episode < self.warmup_episodes:
                action = np.random.choice(self.action_dim)  # explore

            next_state, reward, done, truncated, info = self.env.step(action)
            episode_reward += reward
            steps_in_episode += 1
            self.total_steps += 1

            # add experience to the current episode
            self.append_transition_to_episode(
                episode, state, action, next_state, reward, done
            )

            # update policy network
            if self.total_steps % self.train_frequency == 0:
                loss = self.update_policy_net()
                losses.append(loss)

            # update target network
            if self.total_steps % self.update_frequency == 0:
                self.update_target_net()

            if done:
                break

        # add the episode to the replay buffer
        self.replay_buffer.add_episode(episode)

        end_time = time.time()
        if self.episode > self.warmup_episodes:
            self.explorer.after_episode()  # update epsilon
        avg_loss = np.mean([loss for loss in losses if loss is not None])
        return (
            episode_reward,
            steps_in_episode,
            end_time - start_time,
            avg_loss,
            reset_info,
        )

    def get_next_hidden_state(
        self, state, hidden_state, cell_state
    ) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():  # TODO: check if this is correct. detach?
            current_qs, (new_hidden_state, new_cell_state) = self.policy_net(
                state,
                batch_size=1,
                time_step=1,
                hidden_state=hidden_state,
                cell_state=cell_state,
            )
        self.policy_net.train()  # TODO: check if this is correct
        action = current_qs.argmax().item()
        return action, (new_hidden_state, new_cell_state)

    # update the policy network using a batch of experiences
    # returns the loss for logging
    def update_policy_net(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return

        hidden_batch, cell_batch = self.policy_net.init_hidden_states(
            bsize=self.batch_size
        )
        batch_episodes: List[Episode] = self.replay_buffer.get_batch(
            self.batch_size, self.time_step
        )

        states_batch = []
        actions_batch = []
        next_states_batch = []
        rewards_batch = []
        done_batch = []
        for episode in batch_episodes:
            (
                episode_states,
                episode_actions,
                episode_next_states,
                episode_rewards,
                episode_done,
            ) = zip(*episode)
            states_batch.append(np.asarray(episode_states))
            actions_batch.append(np.asarray(episode_actions))
            next_states_batch.append(np.asarray(episode_next_states))
            rewards_batch.append(np.asarray(episode_rewards))
            done_batch.append(np.asarray(episode_done))

        states_batch_np = np.stack(states_batch)
        actions_batch_np = np.stack(actions_batch)
        next_states_batch_np = np.stack(next_states_batch)
        rewards_batch_np = np.stack(rewards_batch)
        done_batch_np = np.stack(done_batch)

        torch_states_batch = torch.FloatTensor(states_batch_np).to(self.device)
        torch_actions_batch = torch.FloatTensor(actions_batch_np).to(self.device)
        torch_next_states_batch = torch.FloatTensor(next_states_batch_np).to(
            self.device
        )
        torch_rewards_batch = torch.FloatTensor(rewards_batch_np).to(self.device)
        torch_done_batch = torch.FloatTensor(done_batch_np).to(self.device)

        q_values, _ = self.policy_net.forward(
            torch_states_batch,
            self.batch_size,
            self.time_step,
            hidden_batch,
            cell_batch,
        )

        next_q_values, _ = self.target_net.forward(
            torch_next_states_batch,
            self.batch_size,
            self.time_step,
            hidden_batch,
            cell_batch,
        )
        Q_next_max = next_q_values.detach().max(dim=1)[0]
        expected_q_values = (
            torch_rewards_batch[:, self.time_step - 1]
            + (1 - torch_done_batch[:, self.time_step - 1]) * self.gamma * Q_next_max
        )
        q_value = q_values.gather(
            dim=1, index=torch_actions_batch[:, self.time_step - 1].long().unsqueeze(1)
        ).squeeze(1)

        loss = self.loss_fn(q_value, expected_q_values.detach())

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

    def append_transition_to_episode(
        self, episode, state, action, next_state, reward, done
    ):
        episode.append(Transition(state, action, next_state, reward, done))
