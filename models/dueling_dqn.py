# Dueling dqns for sound, vision, and bimodal inputs
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from models.dqn import ReplayBuffer
from wrapper_runners.generic_wrapper_runner import Agent


class DuelingSoundDQN(nn.Module):
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

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()


class DuelingSoundDQNAgent(Agent):
    def load(self, path):
        pass

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
        self.net = DuelingSoundDQN(state_dim, action_dim, hidden_dim).to(device)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=learning_rate, weight_decay=weight_decay
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

    def select_action(self, state, testing, **kwargs):
        epsilon = kwargs["epsilon"]
        if np.random.rand() <= epsilon and not testing:
            # print("random action")
            return np.random.choice(self.action_dim)
        else:
            # print("policy action")
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.net.eval()
            with torch.no_grad():
                q_values = self.net(state)
            self.net.train()
            return q_values.argmax().item()

    def update_model(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return
        state, action, next_state, reward, done = self.replay_buffer.sample(
            self.batch_size
        )
        print(f"sampled next state: {next_state.shape}")
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)

        # state = torch.FloatTensor(state).to(self.device)
        # next_state = torch.FloatTensor(next_state).to(self.device)
        # print(f"{action=}")
        # action = torch.LongTensor(action).to(self.device)
        # reward = torch.FloatTensor(reward).to(self.device)
        # done = torch.FloatTensor(done).to(self.device)
        q_values = self.net(state)
        next_q_values = self.net(next_state)
        q_value = q_values.gather(1, action.to(torch.int64)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        loss = self.loss_fn(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target_model(self):
        pass

    def add_experience(self, state, action, next_state, reward, done):
        print(f"add experience {state.shape=} {next_state.shape=}")
        self.replay_buffer.add(state, action, next_state, reward, done)
