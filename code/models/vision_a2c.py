import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


# Actor module, categorical actions only
class VisionActor(nn.Module):
    def __init__(self, state_dim, n_actions, kernel_size, stride, hidden_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(state_dim[0], 16, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
        )
        conv_out_size = self.get_conv_output(state_dim)
        self.model = nn.Sequential(
            self.conv,
            nn.Flatten(),
            nn.Linear(conv_out_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim=-1),
        )

    def get_conv_output(self, shape):
        x = Variable(torch.rand(1, *shape))
        x = self.conv(x)
        return int(np.prod(x.size()))

    def forward(self, x):
        x = x.float() / 255.0
        return self.model(x)


# Critic module
class VisionCritic(nn.Module):
    def __init__(self, state_dim, kernel_size, stride, hidden_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(state_dim[0], 16, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
        )
        conv_out_size = self.get_conv_output(state_dim)
        self.model = nn.Sequential(
            self.conv,
            nn.Flatten(),
            nn.Linear(conv_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def get_conv_output(self, shape):
        x = Variable(torch.rand(1, *shape))
        x = self.conv(x)
        return int(np.prod(x.size()))

    def forward(self, x):
        x = x.float() / 255.0
        return self.model(x)
