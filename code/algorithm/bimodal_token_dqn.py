from typing import Optional

import torch
from torch import optim

from algorithm.dqn import DQNAlgorithm
from logger import Logger
from models.bimodal_token_replay_buffer import BiModalTokenReplayBuffer
from models.dueling_bimodal_token_dqn import DuelingBiModalTokenDQN


class BimodalTokenDQNAlgorithm(DQNAlgorithm):
    def __init__(
        self,
        env,
        logger: Logger,
        num_episodes: int,
        warmup_episodes: int,
        steps_per_episode: int,
        test_frequency,
        solved_criterion,
        token_dim,
        hidden_dim,
        kernel_size,
        stride,
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
        super().__init__(
            env,
            logger,
            num_episodes,
            warmup_episodes,
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
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.state_dim = env.observation_space["vision"].shape
        self.sound_dim = env.observation_space["sound"].shape
        self.token_dim = token_dim
        self.policy_net = DuelingBiModalTokenDQN(
            self.state_dim,
            self.sound_dim,
            self.token_dim,
            self.action_dim,
            self.kernel_size,
            self.stride,
            self.hidden_dim,
        ).to(device)
        self.target_net = DuelingBiModalTokenDQN(
            self.state_dim,
            self.sound_dim,
            self.token_dim,
            self.action_dim,
            self.kernel_size,
            self.stride,
            self.hidden_dim,
        ).to(device)
        del self.replay_buffer
        self.replay_buffer = BiModalTokenReplayBuffer(replay_buffer_size)
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def add_experience(self, state, action, next_state, reward, done):
        audio = state["sound"]
        video = state["vision"]
        token = state["token"]
        next_audio = next_state["sound"]
        next_video = next_state["vision"]
        next_token = next_state["token"]
        self.replay_buffer.add(
            audio,
            video,
            token,
            action,
            next_audio,
            next_video,
            next_token,
            reward,
            done,
        )

    def exploit_action(self, state) -> int:
        audio_state = state["sound"]
        video_state = state["vision"]
        token_state = state["token"]
        audio_state = torch.FloatTensor(audio_state).unsqueeze(0).to(self.device)
        video_state = torch.FloatTensor(video_state).unsqueeze(0).to(self.device)
        token_state = torch.FloatTensor(token_state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(audio_state, video_state, token_state).detach()
        self.policy_net.train()
        return q_values.argmax().item()

    def update_policy_net(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return
        # print("Will update model")
        (
            audio,
            video,
            token,
            action,
            next_audio,
            next_video,
            next_token,
            reward,
            done,
        ) = self.replay_buffer.sample(self.batch_size)
        audio = audio.to(self.device)
        video = video.to(self.device)
        token = token.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device).squeeze(1)
        next_audio = next_audio.to(self.device)
        next_video = next_video.to(self.device)
        next_token = next_token.to(self.device)
        done = done.to(self.device).squeeze(1)

        q_values = (
            self.policy_net(audio, video, token)
            .gather(1, action.to(torch.int64))
            .squeeze(1)
        )
        next_q_values = self.target_net(next_audio, next_video, next_token).max(1)[0]
        expected_q_values = reward + (1 - done) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, expected_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
