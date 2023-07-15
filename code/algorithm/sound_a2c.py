import numpy as np
import torch
from torch import nn

from algorithm.a2c import A2CAlgorithm
from logger import Logger


# Actor module, categorical actions only
class Actor(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim[0], hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(),
        )

    def forward(self, x):
        return self.model(x)


# Critic module
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim[0], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.model(x)


class SoundA2CAlgorithm(A2CAlgorithm):
    def __init__(
        self,
        env,
        logger: Logger,
        num_episodes: int,
        steps_per_episode: int,
        test_frequency,
        solved_criterion,
        hidden_dim,
        device,
        update_frequency,
        train_frequency,
        batch_size,
        gamma,
        learning_rate,
        weight_decay,
        **kwargs,
    ):
        super().__init__(
            env,
            logger,
            num_episodes,
            steps_per_episode,
            test_frequency,
            solved_criterion,
            hidden_dim,
            device,
            update_frequency,
            train_frequency,
            batch_size,
            gamma,
        )
        self.state_dim = (np.prod(env.observation_space.shape),)
        self.actor = Actor(self.state_dim, self.action_dim, hidden_dim)
        self.critic = Critic(self.state_dim, hidden_dim)
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
