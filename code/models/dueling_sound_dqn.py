# Dueling dqns for sound, vision, and bimodal inputs

import torch
import torch.nn as nn

from code.models.dueling_dqn_base import DuelingDQNAgentBase, DuelingDQNBase
from code.models.replay_buffer import ReplayBuffer


class DuelingSoundDQN(DuelingDQNBase):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DuelingSoundDQN, self).__init__()
        self.feature = nn.Sequential(nn.Linear(state_dim[0], hidden_dim), nn.ReLU())

        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )


class DuelingSoundDQNAgent(DuelingDQNAgentBase):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim,
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
        self.policy_net = DuelingSoundDQN(state_dim, action_dim, hidden_dim).to(device)
        self.target_net = DuelingSoundDQN(state_dim, action_dim, hidden_dim).to(device)
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
            "hidden_dim": self.hidden_dim,
            "buffer_size": self.buffer_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
            "loss_fn": self.loss_fn,
            "weight_decay": self.weight_decay,
        }
