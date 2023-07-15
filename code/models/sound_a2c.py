from torch import nn


# Actor module, categorical actions only
class SoundActor(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim[0], hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.model(x)


# Critic module
class SoundCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.model(x)
