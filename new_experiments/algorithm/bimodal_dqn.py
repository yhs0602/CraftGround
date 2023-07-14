from torch import optim

from models.dueling_bimodal_dqn import DuelingBiModalDQN
from new_experiments.algorithm.dqn import DQNAlgorithm
from new_experiments.logger import Logger


class BimodalDQNAlgorithm(DQNAlgorithm):
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
        self.kernel_size = kernel_size
        self.stride = stride
        self.state_dim = env.observation_space["vision"].shape
        self.sound_dim = env.observation_space["sound"].shape
        self.policy_net = DuelingBiModalDQN(
            self.state_dim,
            self.sound_dim,
            self.action_dim,
            self.kernel_size,
            self.stride,
            self.hidden_dim,
        ).to(device)
        self.target_net = DuelingBiModalDQN(
            self.state_dim,
            self.sound_dim,
            self.action_dim,
            self.kernel_size,
            self.stride,
            self.hidden_dim,
        ).to(device)

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def add_experience(self, state, action, next_state, reward, done):
        self.replay_buffer.add(state, action, next_state, reward, done)
