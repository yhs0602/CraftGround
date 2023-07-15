from typing import Tuple, Optional, List

import numpy as np
import torch
from torch import optim

from code.models.dueling_bimodal_drqn import DuelingBimodalDRQN
from code.models.transition import BimodalTransition, BimodalEpisode
from code.algorithm.drqn import DRQNAlgorithm
from code.new_experiments.logger import Logger


class BimodalDRQNAlgorithm(DRQNAlgorithm):
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
        self.kernel_size = kernel_size
        self.stride = stride
        self.state_dim = env.observation_space["vision"].shape
        self.sound_dim = env.observation_space["sound"].shape

        self.policy_net = DuelingBimodalDRQN(
            self.state_dim,
            self.sound_dim,
            self.action_dim,
            kernel_size,
            stride,
            hidden_dim,
            device,
        ).to(device)
        self.target_net = DuelingBimodalDRQN(
            self.state_dim,
            self.sound_dim,
            self.action_dim,
            kernel_size,
            stride,
            hidden_dim,
            device,
        ).to(device)
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def get_next_hidden_state(
        self, state, hidden_state, cell_state
    ) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        audio_state = state["sound"]
        video_state = state["vision"]
        audio_state = torch.FloatTensor(audio_state).unsqueeze(0).to(self.device)
        video_state = torch.FloatTensor(video_state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():  # TODO: check if this is correct. detach?
            current_qs, (new_hidden_state, new_cell_state) = self.policy_net(
                audio_state,
                video_state,
                batch_size=1,
                time_step=1,
                hidden_state=hidden_state,
                cell_state=cell_state,
            )
        self.policy_net.train()  # TODO: check if this is correct
        action = current_qs.argmax().item()
        return action, (new_hidden_state, new_cell_state)

    def update_policy_net(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return

        hidden_batch, cell_batch = self.policy_net.init_hidden_states(
            bsize=self.batch_size
        )
        batch_episodes: List[BimodalEpisode] = self.replay_buffer.get_batch(
            self.batch_size, self.time_step
        )

        audio_states_batch = []
        video_states_batch = []
        actions_batch = []
        next_audio_states_batch = []
        next_video_states_batch = []
        rewards_batch = []
        done_batch = []
        for episode in batch_episodes:
            (
                episode_audio_states,
                episode_video_states,
                episode_actions,
                episode_next_audio_states,
                episode_next_video_states,
                episode_rewards,
                episode_done,
            ) = zip(*episode)
            audio_states_batch.append(np.asarray(episode_audio_states))
            video_states_batch.append(np.asarray(episode_video_states))
            actions_batch.append(np.asarray(episode_actions))
            next_audio_states_batch.append(np.asarray(episode_next_audio_states))
            next_video_states_batch.append(np.asarray(episode_next_video_states))
            rewards_batch.append(np.asarray(episode_rewards))
            done_batch.append(np.asarray(episode_done))

        audio_states_batch_np = np.stack(audio_states_batch)
        video_states_batch_np = np.stack(video_states_batch)
        actions_batch_np = np.stack(actions_batch)
        next_audio_states_batch_np = np.stack(next_audio_states_batch)
        next_video_states_batch_np = np.stack(next_video_states_batch)
        rewards_batch_np = np.stack(rewards_batch)
        done_batch_np = np.stack(done_batch)

        torch_audio_states_batch = torch.FloatTensor(audio_states_batch_np).to(
            self.device
        )
        torch_video_states_batch = torch.FloatTensor(video_states_batch_np).to(
            self.device
        )
        torch_actions_batch = torch.FloatTensor(actions_batch_np).to(self.device)
        torch_next_audio_states_batch = torch.FloatTensor(
            next_audio_states_batch_np
        ).to(self.device)
        torch_next_video_states_batch = torch.FloatTensor(
            next_video_states_batch_np
        ).to(self.device)
        torch_rewards_batch = torch.FloatTensor(rewards_batch_np).to(self.device)
        torch_done_batch = torch.FloatTensor(done_batch_np).to(self.device)

        q_values, _ = self.policy_net(
            torch_audio_states_batch,
            torch_video_states_batch,
            self.batch_size,
            self.time_step,
            hidden_batch,
            cell_batch,
        )

        next_q_values, _ = self.target_net.forward(
            torch_next_audio_states_batch,
            torch_next_video_states_batch,
            self.batch_size,
            self.time_step,
            hidden_batch,
            cell_batch,
        )
        Q_next_max = next_q_values.detach().max(dim=1)[0]
        expected_q_values = (
            torch_rewards_batch[:, self.time_step - 1]
            + (1 - torch_done_batch[:, self.time_step - 1]) * self.gamma * Q_next_max
        )
        q_value = q_values.gather(
            dim=1, index=torch_actions_batch[:, self.time_step - 1].long().unsqueeze(1)
        ).squeeze(1)

        loss = self.loss_fn(q_value, expected_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def append_transition_to_episode(
        self, episode, state, action, next_state, reward, done
    ):
        audio = state["sound"]
        video = state["vision"]
        next_audio = next_state["sound"]
        next_video = next_state["vision"]
        episode.append(
            BimodalTransition(
                audio, video, action, next_audio, next_video, reward, done
            )
        )
