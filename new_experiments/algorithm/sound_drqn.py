import numpy as np
from torch import optim

from models.dueling_sound_drqn import DuelingSoundDRQN
from new_experiments.algorithm.drqn import DRQNAlgorithm
from new_experiments.logger import Logger


class SoundDRQNAlgorithm(DRQNAlgorithm):
    def __init__(
        self,
        env,
        logger: Logger,
        num_episodes: int,
        warmup_episodes: int,
        steps_per_episode: int,
        test_frequency,
        solved_criterion,
        hidden_dim,
        device,
        epsilon_init,
        epsilon_decay,
        epsilon_min,
        update_frequency,
        train_frequency,
        replay_buffer_size,
        batch_size,
        time_step,
        gamma,
        learning_rate,
        weight_decay,
        tau,
        **kwargs
    ):
        super().__init__(
            env,
            logger,
            num_episodes,
            warmup_episodes,
            steps_per_episode,
            test_frequency,
            solved_criterion,
            hidden_dim,
            device,
            epsilon_init,
            epsilon_decay,
            epsilon_min,
            update_frequency,
            train_frequency,
            replay_buffer_size,
            batch_size,
            time_step,
            gamma,
            learning_rate,
            weight_decay,
            tau,
        )

        self.state_dim = (np.prod(env.observation_space.shape),)
        self.policy_net = DuelingSoundDRQN(
            self.state_dim, self.action_dim, hidden_dim, device
        ).to(device)
        self.target_net = DuelingSoundDRQN(
            self.state_dim, self.action_dim, hidden_dim, device
        ).to(device)
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
