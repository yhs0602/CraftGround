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

        vision_states_batch_np = np.stack([state["vision"] for state in states])
        audio_states_batch_np = np.stack([state["sound"] for state in states])
        actions_batch_np = np.stack(actions)
        rewards_batch_np = np.stack(rewards)
        next_vision_states_batch_np = np.stack(
            [state["vision"] for state in next_states]
        )
        next_audio_states_batch_np = np.stack([state["sound"] for state in next_states])
        dones_batch_np = np.stack(dones)

        vision_states_batch = torch.FloatTensor(vision_states_batch_np).to(self.device)
        audio_states_batch = torch.FloatTensor(audio_states_batch_np).to(self.device)
        actions_batch = torch.LongTensor(actions_batch_np).to(self.device)
        rewards_batch = torch.FloatTensor(rewards_batch_np).to(self.device)
        next_vision_states_batch = torch.FloatTensor(next_vision_states_batch_np).to(
            self.device
        )
        next_audio_states_batch = torch.FloatTensor(next_audio_states_batch_np).to(
            self.device
        )
        dones_batch = torch.FloatTensor(dones_batch_np).to(self.device)

        probs, value = self.actor_critic(audio_states_batch, vision_states_batch)
        _, next_value = self.actor_critic(
            next_audio_states_batch, next_vision_states_batch
        )

        advantage = rewards_batch + self.gamma * next_value * (1 - dones_batch) - value
        dist = torch.distributions.Categorical(probs=probs)
        actor_loss = -dist.log_prob(actions_batch) * advantage.detach()
        critic_loss = advantage.pow(2)
        loss = actor_loss + critic_loss

        self.actor_critic_optim.zero_grad()
        loss.mean().backward()
        self.actor_critic_optim.step()

        time_took = time.time() - start_time

        avg_actor_loss = actor_loss.mean().item()
        avg_critic_loss = critic_loss.mean().item()
        action_entropy = dist.entropy().mean().item()

        return (
            episode_reward,
            steps_in_episode,
            time_took,
            avg_actor_loss,
            avg_critic_loss,
            reset_info,
            action_entropy,
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
