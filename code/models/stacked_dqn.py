from typing import Optional

import numpy as np
import torch
from gymnasium.wrappers import LazyFrames
from torch import optim, nn

from code.models.dqn import SoundDQN, ReplayBuffer, DQNAgent


class StackedDQNSoundAgent(DQNAgent):
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
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.stack_size = stack_size
        self.device = device
        self.policy_net = SoundDQN(state_dim, action_dim, hidden_dim).to(device)
        self.target_net = SoundDQN(state_dim, action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

    @property
    def config(self):
        return {
            "architecture": "Stacked Sound DQN",
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
            "stack_size": self.stack_size,
        }

    def select_action(self, state, testing, **kwargs):
        epsilon = kwargs["epsilon"]
        if np.random.rand() <= epsilon and not testing:
            # print("random action")
            return np.random.choice(self.action_dim)
        else:
            # print("policy action")
            state = torch.FloatTensor(state.__array__()).unsqueeze(0).to(self.device)
            self.policy_net.eval()
            with torch.no_grad():
                q_values = self.policy_net(state)
            self.policy_net.train()
            return q_values.argmax().item()

    def update_model(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return
        # print("Will update model")
        state, action, next_state, reward, done = self.replay_buffer.sample(
            self.batch_size
        )
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device).squeeze(1)
        next_state = next_state.to(self.device)
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

    def add_experience(self, state: LazyFrames, action, next_state, reward, done):
        # state and next_state is lazy
        self.replay_buffer.add(
            state.__array__(np.float32),
            action,
            next_state.__array__(np.float32),
            reward,
            done,
        )
