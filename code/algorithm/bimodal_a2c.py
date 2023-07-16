import time

import numpy as np
import torch
import torch.optim

from algorithm.a2c import A2CAlgorithm
from logger import Logger
from models.bimodal_a2c import BimodalActorCritic


class BimodalA2CAlgorithm(A2CAlgorithm):
    def __init__(
        self,
        env,
        logger: Logger,
        num_episodes: int,
        steps_per_episode: int,
        test_frequency,
        solved_criterion,
        hidden_dim,
        kernel_size,
        stride,
        device,
        update_frequency,
        train_frequency,
        gamma,
        optimizer,
        **kwargs,
    ):
        super().__init__(
            env,
            logger,
            num_episodes,
            steps_per_episode,
            test_frequency,
            solved_criterion,
            hidden_dim,
            device,
            update_frequency,
            train_frequency,
            gamma,
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.state_dim = env.observation_space["vision"].shape
        self.sound_dim = env.observation_space["sound"].shape
        self.actor_critic = BimodalActorCritic(
            self.state_dim,
            self.sound_dim,
            self.action_dim,
            kernel_size,
            stride,
            hidden_dim,
        ).to(device)
        optim_name = optimizer.get("name", "Adam")
        optimizer_class = getattr(torch.optim, optim_name)
        self.actor_critic_optim = optimizer_class(
            self.actor_critic.parameters(), **optimizer["params"]
        )

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
            _, action, _ = self.exploit_action(state)
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

        states, actions, rewards, next_states, dones = [], [], [], [], []

        for step in range(self.steps_per_episode):
            self.logger.before_step(step, should_record_video=False)
            dist, action, advantages = self.exploit_action(state)
            action = action.item()
            next_state, reward, done, truncated, info = self.env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            state = next_state
            episode_reward += reward
            steps_in_episode += 1
            if done:
                break

        loss = 0
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            next_state = next_states[i]
            done = dones[i]
            audio_state = state["sound"]
            video_state = state["vision"]
            audio_state = torch.FloatTensor(audio_state).unsqueeze(0).to(self.device)
            video_state = torch.FloatTensor(video_state).unsqueeze(0).to(self.device)
            next_audio_state = next_state["sound"]
            next_video_state = next_state["vision"]
            next_audio_state = (
                torch.FloatTensor(next_audio_state).unsqueeze(0).to(self.device)
            )
            next_video_state = (
                torch.FloatTensor(next_video_state).unsqueeze(0).to(self.device)
            )
            action = torch.LongTensor([action]).unsqueeze(0).to(self.device)
            reward = torch.FloatTensor([reward]).unsqueeze(0).to(self.device)
            probs, value = self.actor_critic(audio_state, video_state)
            _, next_value = self.actor_critic(next_audio_state, next_video_state)

            advantage = reward + self.gamma * next_value * (1 - done) - value
            dist = torch.distributions.Categorical(probs=probs)
            actor_loss = -dist.log_prob(action) * advantage.detach()
            critic_loss = advantage.pow(2)
            loss += actor_loss + critic_loss

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        self.actor_critic_optim.zero_grad()
        loss.backward()
        self.actor_critic_optim.step()

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
        audio_state = state["sound"]
        video_state = state["vision"]
        audio_state = torch.FloatTensor(audio_state).unsqueeze(0).to(self.device)
        video_state = torch.FloatTensor(video_state).unsqueeze(0).to(self.device)
        probs, advantages = self.actor_critic(audio_state, video_state)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        return dist, action, advantages
