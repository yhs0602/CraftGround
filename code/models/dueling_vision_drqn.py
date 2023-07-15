# Dueling drqn for vision
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


# https://github.com/keep9oing/DRQN-Pytorch-CartPole-v1/blob/main/DRQN.py
class DuelingVisionDRQN(nn.Module):
    def __init__(self, state_dim, action_dim, kernel_size, stride, hidden_dim, device):
        super(DuelingVisionDRQN, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.state_dim = state_dim
        self.feature = nn.Sequential(
            nn.Conv2d(state_dim[0], 16, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
        )
        conv_out_size = self.get_conv_output(state_dim)
        self.lstm = nn.LSTM(
            input_size=conv_out_size,
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
        x = self.feature(x)
        return int(np.prod(x.size()))

    def forward(
        self, x, batch_size, time_step, hidden_state, cell_state
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = x.view(batch_size * time_step, *self.state_dim)
        x = x.float() / 255.0
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
