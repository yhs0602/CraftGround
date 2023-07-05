import random
from collections import deque
from typing import List, Optional, TypeAlias

import numpy as np
import torch
from torch import nn

from final_experiments.wrapper_runners.generic_wrapper_runner import Agent
from models.dqn import Transition

# https://github.com/mynkpl1998/Recurrent-Deep-Q-Learning/blob/master/LSTM%2C%20BPTT%3D8.ipynb

Episode: TypeAlias = List[Transition]


class RecurrentReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def add_episode(self, episode: Episode):
        self.memory.append(episode)

    def get_batch(
        self, batch_size, time_step
    ) -> List[Episode]:  # Actually, partial episode
        sampled_episodes = random.sample(self.memory, batch_size)
        batch = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - time_step)
            batch.append(episode[point : point + time_step])
        return batch

    def __len__(self):
        return len(self.memory)


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
        x = x.view(batch_size, time_step, -1)
        x = self.feature(x)
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


class DuelingSoundDRQNAgent(Agent):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim,
        buffer_size,
        batch_size,
        time_step,
        gamma,
        learning_rate,
        weight_decay,
        device,
        stack_size=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.time_step = time_step
        self.device = device
        self.policy_net = DuelingSoundDRQN(
            state_dim, action_dim, hidden_dim, device=device
        ).to(device)
        self.target_net = DuelingSoundDRQN(
            state_dim, action_dim, hidden_dim, device=device
        ).to(device)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.buffer_size = buffer_size
        self.replay_buffer = RecurrentReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    @property
    def config(self):
        return {
            "architecture": "Dueling sound DRQN",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "buffer_size": self.buffer_size,
            "time_step": self.time_step,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
            "loss_fn": self.loss_fn,
            "weight_decay": self.weight_decay,
        }

    def load(self, path):
        pass

    def select_action(self, state, testing, **kwargs):
        epsilon = kwargs["epsilon"]
        hidden_state = kwargs["hidden_state"]
        cell_state = kwargs["cell_state"]
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()  # TODO: check if this is correct
        with torch.no_grad():  # TODO: check if this is correct. detach?
            current_qs, (new_hidden_state, new_cell_state) = self.policy_net(
                state,
                batch_size=1,
                time_step=1,
                hidden_state=hidden_state,
                cell_state=cell_state,
            )
        self.policy_net.train()  # TODO: check if this is correct
        if np.random.rand() <= epsilon and not testing:
            # print("random action")
            action = np.random.choice(self.action_dim)
            return action, (new_hidden_state, new_cell_state)
        else:
            # print("policy action")
            action = current_qs.argmax().item()
            return action, (new_hidden_state, new_cell_state)

    def update_model(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return

        hidden_batch, cell_batch = self.policy_net.init_hidden_states(
            bsize=self.batch_size
        )
        batch_episodes: List[Episode] = self.replay_buffer.get_batch(
            self.batch_size, self.time_step
        )

        states_batch = []
        actions_batch = []
        next_states_batch = []
        rewards_batch = []
        done_batch = []
        for episode in batch_episodes:
            states = []
            actions = []
            next_states = []
            rewards = []
            done = []
            for transition in episode:
                states.append(transition.state)
                actions.append(transition.action)
                next_states.append(transition.next_state)
                rewards.append(transition.reward)
                done.append(transition.done)
            states_batch.append(states)
            actions_batch.append(actions)
            next_states_batch.append(next_states)
            rewards_batch.append(rewards)
            done_batch.append(done)

        torch_states_batch = torch.FloatTensor(states_batch).to(self.device)
        torch_actions_batch = torch.FloatTensor(actions_batch).to(self.device)
        torch_next_states_batch = torch.FloatTensor(next_states_batch).to(self.device)
        torch_rewards_batch = torch.FloatTensor(rewards_batch).to(self.device)
        torch_done_batch = torch.FloatTensor(done_batch).to(self.device)

        q_values, _ = self.policy_net.forward(
            torch_states_batch,
            self.batch_size,
            self.time_step,
            hidden_batch,
            cell_batch,
        )

        next_q_values, _ = self.target_net.forward(
            torch_next_states_batch,
            self.batch_size,
            self.time_step,
            hidden_batch,
            cell_batch,
        )
        Q_next_max = next_q_values.detach().max(dim=1)[0]
        expected_q_values = (
            torch_rewards_batch[:, self.time_step - 1]
            + (1 - torch_done_batch[:, self.time_step - 1]) * self.gamma * Q_next_max
        )
        q_value = q_values.gather(
            dim=1, index=torch_actions_batch[:, self.time_step - 1].long().unsqueeze(1)
        ).squeeze(1)

        loss = self.loss_fn(q_value, expected_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def add_episode(self, episode: Episode):
        self.replay_buffer.add_episode(episode)
