# Dueling dqns for sound, vision, and bimodal inputs
from typing import Optional

import torch
import torch.nn as nn

from models.dqn import Transition
from models.dueling_dqn_base import DuelingDQNAgentBase
from models.dueling_sound_dqn import DuelingSoundDQN
from models.per import PER
import torch.nn.functional as F


class PERDuelingSoundDQNAgent(DuelingDQNAgentBase):
    replay_buffer: PER

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
        self.replay_buffer = PER(buffer_size)
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    @property
    def config(self):
        return {
            "architecture": "Prioritized Dueling Sound DQN",
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

    def add_experience(self, state, action, next_state, reward, done):
        self.policy_net.eval()
        with torch.no_grad():
            old_policy_values = self.policy_net(
                torch.FloatTensor(state).unsqueeze(0).to(self.device)
            )
            old_val = old_policy_values[0][action]
        self.policy_net.train()

        self.target_net.eval()
        with torch.no_grad():
            target_q_values = self.target_net(
                torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            )
        self.target_net.train()

        if done:
            old_policy_values[0][action] = reward
        else:
            old_policy_values[0][action] = reward + self.gamma * torch.max(
                target_q_values
            )
        error = abs(old_val - old_policy_values[0][action])
        self.replay_buffer.add(
            error.item(), Transition(state, action, next_state, reward, done)
        )

    def update_model(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return
        transitions, idxs, is_weights = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state = torch.stack(list(map(torch.from_numpy, batch.state)))
        action = torch.stack([torch.Tensor([float(x)]) for x in batch.action])
        reward = torch.stack([torch.Tensor([float(x)]) for x in batch.reward])
        next_state = torch.stack(list(map(torch.from_numpy, batch.next_state)))
        done = torch.stack([torch.Tensor([x]) for x in batch.done])

        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device).squeeze(1)
        done = done.to(self.device).squeeze(1)

        q_values = self.policy_net(state).gather(1, action.to(torch.int64)).squeeze(1)
        next_q_values = self.target_net(next_state).max(1)[0]
        expected_q_values = reward + (1 - done) * self.gamma * next_q_values

        errors = torch.abs(q_values - expected_q_values.detach()).detach().cpu().numpy()

        for i in range(self.batch_size):
            idx = idxs[i]
            self.replay_buffer.update(idx, errors[i])

        loss = (
            torch.FloatTensor(is_weights).to(self.device)
            * F.mse_loss(q_values, expected_q_values.detach())
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
