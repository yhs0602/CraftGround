import torch
from torch import nn


# https://github.com/mynkpl1998/Recurrent-Deep-Q-Learning/blob/master/LSTM%2C%20BPTT%3D8.ipynb


class DuelingSoundDRQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, device):
        super(DuelingSoundDRQN, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.feature = nn.Sequential(nn.Linear(state_dim[0], hidden_dim), nn.ReLU())
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
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

    def forward(self, x, batch_size, time_step, hidden_state, cell_state):
        x = x.view(batch_size * time_step, -1)
        x = self.feature(x)
        x = x.view(batch_size, time_step, -1)
        # print(f"1 {x.shape=} {hidden_state.shape=} {cell_state.shape=}")
        x, (hidden_state, cell_state) = self.lstm(x, (hidden_state, cell_state))
        # print(f"2 {x.shape=} {hidden_state.shape=} {cell_state.shape=}")
        x = x[:, -1, :]
        # print(f"3 {x.shape=} {hidden_state.shape=} {cell_state.shape=}")
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(), (hidden_state, cell_state)

    def init_hidden_states(self, bsize):
        h = torch.zeros(1, bsize, self.hidden_dim).float().to(self.device)
        c = torch.zeros(1, bsize, self.hidden_dim).float().to(self.device)
        return h, c
