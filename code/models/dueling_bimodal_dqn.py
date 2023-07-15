# Dueling dqns for sound, vision, and bimodal inputs
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from code.models.bimodal_replay_buffer import BiModalReplayBuffer
from code.models.dueling_dqn_base import DuelingDQNAgentBase, DuelingDQNBase


class DuelingBiModalDQN(DuelingDQNBase):
    def __init__(
        self, state_dim, sound_dim, action_dim, kernel_size, stride, hidden_dim
    ):
        super(DuelingBiModalDQN, self).__init__()
        self.audio_feature = nn.Sequential(
            nn.Linear(sound_dim[0], hidden_dim), nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(state_dim[0], 16, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
        )
        conv_out_size = self.get_conv_output(state_dim)
        self.video_feature = nn.Sequential(
            self.conv, nn.Flatten(), nn.Linear(conv_out_size, hidden_dim), nn.ReLU()
        )
        self.feature = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def get_conv_output(self, shape):
        x = Variable(torch.rand(1, *shape))
        x = self.conv(x)
        return int(np.prod(x.size()))

    def forward(self, audio, video):
        video = video.float() / 255.0
        audio_feature = self.audio_feature(audio)
        video_feature = self.video_feature(video)
        x = torch.cat((audio_feature, video_feature), dim=1)
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DuelingBiModalDQNAgent(DuelingDQNAgentBase):
    replay_buffer: BiModalReplayBuffer  # override type hint

    def __init__(
        self,
        state_dim,
        sound_dim,
        action_dim,
        hidden_dim,
        kernel_size,
        stride,
        buffer_size,
        batch_size,
        gamma,
        learning_rate,
        weight_decay,
        device,
        stack_size=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.policy_net = DuelingBiModalDQN(
            state_dim, sound_dim, action_dim, kernel_size, stride, hidden_dim
        ).to(device)
        self.target_net = DuelingBiModalDQN(
            state_dim, sound_dim, action_dim, kernel_size, stride, hidden_dim
        ).to(device)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.buffer_size = buffer_size
        self.replay_buffer = MultiModalReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    @property
    def config(self):
        return {
            "architecture": "Dueling DQN",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "buffer_size": self.buffer_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
            "loss_fn": self.loss_fn,
            "weight_decay": self.weight_decay,
        }

    def select_action(self, state, testing, **kwargs):
        epsilon = kwargs["epsilon"]
        if np.random.rand() <= epsilon and not testing:
            # print("random action")
            return np.random.choice(self.action_dim)
        else:
            # print("policy action")
            audio_state = state["sound"]
            video_state = state["vision"]
            audio_state = torch.FloatTensor(audio_state).unsqueeze(0).to(self.device)
            video_state = torch.FloatTensor(video_state).unsqueeze(0).to(self.device)
            self.policy_net.eval()
            with torch.no_grad():
                q_values = self.policy_net(audio_state, video_state).detach()
            self.policy_net.train()
            return q_values.argmax().item()

    def update_model(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return
        # print("Will update model")
        (
            audio,
            video,
            action,
            next_audio,
            next_video,
            reward,
            done,
        ) = self.replay_buffer.sample(self.batch_size)
        audio = audio.to(self.device)
        video = video.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device).squeeze(1)
        next_audio = next_audio.to(self.device)
        next_video = next_video.to(self.device)
        done = done.to(self.device).squeeze(1)

        q_values = (
            self.policy_net(audio, video).gather(1, action.to(torch.int64)).squeeze(1)
        )
        next_q_values = self.target_net(next_audio, next_video).max(1)[0]
        expected_q_values = reward + (1 - done) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, expected_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def add_experience(self, state, action, next_state, reward, done):
        audio = state["sound"]
        video = state["vision"]
        next_audio = next_state["sound"]
        next_video = next_state["vision"]
        self.replay_buffer.add(
            audio, video, action, next_audio, next_video, reward, done
        )
