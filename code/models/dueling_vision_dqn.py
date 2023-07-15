# Dueling dqns for sound, vision, and bimodal inputs

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from code.models.dueling_dqn_base import DuelingDQNBase


class DuelingVisionDQN(DuelingDQNBase):
    def __init__(self, state_dim, action_dim, kernel_size, stride, hidden_dim):
        super(DuelingVisionDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(state_dim[0], 16, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
        )
        conv_out_size = self.get_conv_output(state_dim)
        self.advantage = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def get_conv_output(self, shape):
        x = Variable(torch.rand(1, *shape))
        x = self.feature(x)
        return int(np.prod(x.size()))

    def forward(self, x):
        x = x.float() / 255.0
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
