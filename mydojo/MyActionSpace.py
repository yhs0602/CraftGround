from typing import Optional

import gymnasium as gym
import numpy as np


class MyActionSpace(gym.spaces.Discrete):
    def __init__(self, n):
        super(MyActionSpace, self).__init__(n)

    def sample(self, mask: Optional[np.ndarray] = None) -> int:
        super(MyActionSpace, self).sample()
        return np.random.randint(self.n)

    def contains(self, x):
        return x in range(self.n)

    def __repr__(self):
        return "MyActionSpace(%d)" % self.n

    def __eq__(self, other):
        return self.n == other.n
