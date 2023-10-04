from typing import Optional

import gymnasium as gym
import numpy as np


class ActionSpace(gym.spaces.Discrete):
    def __init__(self, n):
        super(ActionSpace, self).__init__(n)

    def sample(self, mask: Optional[np.ndarray] = None) -> int:
        super(ActionSpace, self).sample()
        return np.random.randint(self.n)

    def contains(self, x):
        return x in range(self.n)

    def __repr__(self):
        return "MyActionSpace(%d)" % self.n

    def __eq__(self, other):
        return self.n == other.n


class MultiActionSpace(gym.spaces.MultiDiscrete):
    def __init__(self, nvec: list[int]):
        super(MultiActionSpace, self).__init__(nvec)

    # def sample(self, mask: Optional[np.ndarray] = None) -> NDArray[np.integer[Any]]:
    #     return super(MultiActionSpace, self).sample()

    def __repr__(self):
        return "MultiActionSpace(%d)" % self.nvec
