# Dueling dqns for sound, vision, and bimodal inputs

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from models.dqn import ReplayBuffer
from models.dueling_dqn_base import DuelingDQNBase, DuelingDQNAgentBase


class DuelingVisionRNNDQN(DuelingDQNBase):
    def __init__(self, state_dim, action_dim, kernel_size, stride, hidden_dim):
        super(DuelingVisionRNNDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(state_dim[0], 16, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
        )
        conv_out_size = self.get_conv_output(state_dim)
        self.rnn = nn.GRUCell(conv_out_size, hidden_dim)
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

    def forward(self, x, hx):
        x = x.float() / 255.0
        x = self.feature(x)
        x = x.view(x.size(0), -1)

        hx = self.rnn(x, hx)
        x = nn.functional.relu(hx)

        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DuelingVisionDQNAgent(DuelingDQNAgentBase):
    def __init__(
        self,
        state_dim,
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
        self.kernel_size = kernel_size
        self.stride = stride
        self.device = device
        self.policy_net = DuelingVisionRNNDQN(
            state_dim, action_dim, kernel_size, stride, hidden_dim
        ).to(device)
        self.target_net = DuelingVisionRNNDQN(
            state_dim, action_dim, kernel_size, stride, hidden_dim
        ).to(device)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBuffer(buffer_size)
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
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "hidden_dim": self.hidden_dim,
            "buffer_size": self.buffer_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
            "loss_fn": self.loss_fn,
            "weight_decay": self.weight_decay,
        }
