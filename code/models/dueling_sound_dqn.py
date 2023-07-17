# Dueling dqns for sound, vision, and bimodal inputs

import torch.nn as nn

from models.dueling_dqn_base import DuelingDQNBase


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
