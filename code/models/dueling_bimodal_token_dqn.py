# Dueling dqns for sound, vision, and bimodal inputs

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from models.dueling_dqn_base import DuelingDQNBase


class DuelingBiModalTokenDQN(DuelingDQNBase):
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
        super(DuelingBiModalTokenDQN, self).__init__()
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
        video = video.float() / 255.0
        audio_feature = self.audio_feature(audio)
        video_feature = self.video_feature(video)
        token_feature = token
        x = torch.cat((audio_feature, video_feature, token_feature), dim=1)
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
