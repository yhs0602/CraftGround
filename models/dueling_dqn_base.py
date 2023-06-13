# Dueling dqns for sound, vision, and bimodal inputs
from abc import ABC
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from models.dqn import ReplayBuffer
from wrapper_runners.generic_wrapper_runner import Agent


class DuelingDQNBase(nn.Module, ABC):
    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DuelingDQNAgentBase(Agent, ABC):
    action_dim: Tuple[int, ...]
    policy_net: DuelingDQNBase
    target_net: DuelingDQNBase
    device: torch.device
    replay_buffer: ReplayBuffer
    batch_size: int
    loss_fn: _Loss
    gamma: float
    optimizer: torch.optim.Optimizer

    def load(self, path):
        pass

    def select_action(self, state, testing, **kwargs):
        epsilon = kwargs["epsilon"]
        if np.random.rand() <= epsilon and not testing:
            # print("random action")
            return np.random.choice(self.action_dim)
        else:
            # print("policy action")
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.policy_net.eval()
            with torch.no_grad():
                q_values = self.policy_net(state).detach()
            self.policy_net.train()
            return q_values.argmax().item()

    def update_model(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return
        state, action, next_state, reward, done = self.replay_buffer.sample(
            self.batch_size
        )
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device).squeeze(1)
        done = done.to(self.device).squeeze(1)

        q_values = self.policy_net(state).gather(1, action.to(torch.int64)).squeeze(1)
        next_q_values = self.target_net(next_state).max(1)[0]
        expected_q_values = reward + (1 - done) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, expected_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def add_experience(self, state, action, next_state, reward, done):
        self.replay_buffer.add(state, action, next_state, reward, done)
