import random
from collections import deque

import torch

from models.transition import BimodalTokenTransition


class BiModalTokenReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def add(self, *args):
        self.memory.append(BimodalTokenTransition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = BimodalTokenTransition(*zip(*transitions))
        audio = torch.stack(list(map(torch.from_numpy, batch.audio)))
        video = torch.stack(list(map(torch.from_numpy, batch.video)))
        token = torch.stack(list(map(torch.from_numpy, batch.token)))
        action = torch.stack([torch.Tensor([float(x)]) for x in batch.action])
        reward = torch.stack([torch.Tensor([float(x)]) for x in batch.reward])
        next_audio = torch.stack(list(map(torch.from_numpy, batch.next_audio)))
        next_video = torch.stack(list(map(torch.from_numpy, batch.next_video)))
        next_token = torch.stack(list(map(torch.from_numpy, batch.next_token)))
        done = torch.stack([torch.Tensor([x]) for x in batch.done])
        return (
            audio,
            video,
            token,
            action,
            next_audio,
            next_video,
            next_token,
            reward,
            done,
        )  # tuple(map(torch.cat, batch))
