import random
from collections import deque

import torch

from models.transition import BimodalTransition


class BiModalReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def add(self, *args):
        self.memory.append(BimodalTransition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = BimodalTransition(*zip(*transitions))
        audio = torch.stack(list(map(torch.from_numpy, batch.audio)))
        video = torch.stack(list(map(torch.from_numpy, batch.video)))
        action = torch.stack([torch.Tensor([float(x)]) for x in batch.action])
        reward = torch.stack([torch.Tensor([float(x)]) for x in batch.reward])
        next_audio = torch.stack(list(map(torch.from_numpy, batch.next_audio)))
        next_video = torch.stack(list(map(torch.from_numpy, batch.next_video)))
        done = torch.stack([torch.Tensor([x]) for x in batch.done])
        return (
            audio,
            video,
            action,
            next_audio,
            next_video,
            reward,
            done,
        )  # tuple(map(torch.cat, batch))
