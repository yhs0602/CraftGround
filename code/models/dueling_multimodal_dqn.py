# Dueling dqns for sound, vision, and bimodal inputs

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from models.dueling_dqn_base import DuelingDQNBase


class DuelingMultimodalDQN(DuelingDQNBase):
    def __init__(
        self,
        state_dim,
        sound_dim,
        token_dim,
        action_dim,
        kernel_size,
        stride,
        hidden_dim,
    ):
        super(DuelingMultimodalDQN, self).__init__()
        self.state_dim = state_dim
        self.sound_dim = sound_dim
        self.token_dim = token_dim
        self.action_dim = action_dim
        if sound_dim > 0:
            self.audio_feature = nn.Sequential(
                nn.Linear(sound_dim[0], hidden_dim), nn.ReLU()
            )
        if state_dim:
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
            nn.Linear(hidden_dim * 2 + token_dim, hidden_dim),
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

    def forward(self, audio, video, token):
        features = []
        if self.state_dim:
            video = video.float() / 255.0
            video_feature = self.video_feature(video)
            features.append(video_feature)
        if self.sound_dim > 0:
            audio_feature = self.audio_feature(audio)
            features.append(audio_feature)
        if self.token_dim > 0:
            token_feature = token
            features.append(token_feature)
        x = torch.cat(features, dim=1)
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
