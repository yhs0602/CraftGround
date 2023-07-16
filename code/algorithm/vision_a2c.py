import torch
import torch.optim

from algorithm.a2c import A2CAlgorithm
from logger import Logger
from models.vision_a2c import VisionActor, VisionCritic


class VisionA2CAlgorithm(A2CAlgorithm):
    def __init__(
        self,
        env,
        logger: Logger,
        num_episodes: int,
        steps_per_episode: int,
        test_frequency,
        solved_criterion,
        hidden_dim,
        kernel_size,
        stride,
        device,
        update_frequency,
        train_frequency,
        gamma,
        optimizer,
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
            gamma,
        )
        self.state_dim = env.observation_space.shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.actor = VisionActor(
            self.state_dim, self.action_dim, kernel_size, stride, hidden_dim
        ).to(device)
        self.critic = VisionCritic(self.state_dim, kernel_size, stride, hidden_dim).to(
            device
        )
        optim_name = optimizer.get("name", "Adam")
        optimizer_class = getattr(torch.optim, optim_name)
        self.actor_optim = optimizer_class(
            self.actor.parameters(), **optimizer["params"]
        )
        self.critic_optim = optimizer_class(
            self.actor.parameters(), **optimizer["params"]
        )
