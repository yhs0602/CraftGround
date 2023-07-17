from torch import optim

from models.dueling_vision_dqn import DuelingVisionDQN
from algorithm.dqn import DQNAlgorithm
from logger import Logger


class VisionDQNAlgorithm(DQNAlgorithm):
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
        kernel_size,
        stride,
        device,
        epsilon_init,
        epsilon_decay,
        epsilon_min,
        update_frequency,
        train_frequency,
        replay_buffer_size,
        batch_size,
        gamma,
        learning_rate,
        weight_decay,
        tau,
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
            gamma,
            learning_rate,
            weight_decay,
            tau,
        )
        self.state_dim = env.observation_space.shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.policy_net = DuelingVisionDQN(
            self.state_dim, self.action_dim, kernel_size, stride, hidden_dim
        ).to(device)
        self.target_net = DuelingVisionDQN(
            self.state_dim, self.action_dim, kernel_size, stride, hidden_dim
        ).to(device)

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
