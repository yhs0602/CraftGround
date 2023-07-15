import numpy as np


class EpsilonGreedyExplorer:
    def __init__(self, epsilon, decay, min_epsilon):
        self.epsilon = epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon

    def should_explore(self):
        return np.random.random() < self.epsilon

    def after_episode(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
