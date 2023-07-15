# https://github.com/rlcode/per/blob/master/prioritized_memory.py
import random
from typing import Tuple, List, TypeVar, Generic

import numpy as np

from code.models.sumtree import SumTree

T = TypeVar("T")  # Declare type variable "T"


class PER(Generic[T]):  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def __len__(self):
        return self.tree.n_entries

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample: T):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n) -> Tuple[List[T], List[int], List[float]]:
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
