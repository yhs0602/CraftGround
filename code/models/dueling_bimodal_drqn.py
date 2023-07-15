# Dueling drqn for bimodal inputs
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


# https://github.com/keep9oing/DRQN-Pytorch-CartPole-v1/blob/main/DRQN.py
class DuelingBimodalDRQN(nn.Module):
    def __init__(
        self, state_dim, sound_dim, action_dim, kernel_size, stride, hidden_dim, device
    ):
        super(DuelingBimodalDRQN, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sound_dim = sound_dim
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

        self.feature = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU())

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
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

    def forward(
        self, audio, video, batch_size, time_step, hidden_state, cell_state
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        video = video.view(batch_size * time_step, *self.state_dim)
        video = video.float() / 255.0
        video_feature = self.video_feature(video)
        audio = audio.view(batch_size * time_step, -1)
        audio_feature = self.audio_feature(audio)
        x = torch.cat((audio_feature, video_feature), dim=1)
        x = self.feature(x)
        x = x.view(batch_size, time_step, -1)
        x, (hidden_state, cell_state) = self.lstm(x, (hidden_state, cell_state))
        x = x[:, -1, :]
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(), (hidden_state, cell_state)

    def init_hidden_states(self, bsize):
        h = torch.zeros(1, bsize, self.hidden_dim).float().to(self.device)
        c = torch.zeros(1, bsize, self.hidden_dim).float().to(self.device)
        return h, c
