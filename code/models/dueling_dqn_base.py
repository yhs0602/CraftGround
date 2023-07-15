# Dueling dqns for sound, vision, and bimodal inputs
from abc import ABC

import torch.nn as nn


class DuelingDQNBase(nn.Module, ABC):
    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
