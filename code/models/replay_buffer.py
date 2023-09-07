import random
from collections import deque

import torch

from models.transition import Transition


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def add(self, *args):
        state, action, next_state, reward, done = args
        t_state = torch.from_numpy(state)
        t_action = torch.Tensor([float(action)])
        t_next_state = torch.from_numpy(next_state)
        t_reward = torch.Tensor([float(reward)])
        t_done = torch.Tensor([done])
        self.memory.append(
            Transition(t_state, t_action, t_next_state, t_reward, t_done)
        )

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return (
            torch.stack(batch.state),
            torch.stack(batch.action),
            torch.stack(batch.next_state),
            torch.stack(batch.reward),
            torch.stack(batch.done),
        )
