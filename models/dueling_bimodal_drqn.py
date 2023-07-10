# Dueling drqn for bimodal inputs
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from final_experiments.wrapper_runners.generic_wrapper_runner import Agent
from models.recurrent_replay_buffer import Episode, RecurrentReplayBuffer
from models.transition import BimodalEpisode


# https://github.com/keep9oing/DRQN-Pytorch-CartPole-v1/blob/main/DRQN.py
class DuelingBimodalDRQN(nn.Module):
    def __init__(
        self, state_dim, sound_dim, action_dim, kernel_size, stride, hidden_dim, device
    ):
        super(DuelingBimodalDRQN, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sound_dim = sound_dim
        self.audio_feature = nn.Sequential(
            nn.Linear(sound_dim[0], hidden_dim), nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(state_dim[0], 16, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
        )
        conv_out_size = self.get_conv_output(state_dim)
        self.video_feature = nn.Sequential(
            self.conv, nn.Flatten(), nn.Linear(conv_out_size, hidden_dim), nn.ReLU()
        )

        self.feature = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU())

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def get_conv_output(self, shape):
        x = Variable(torch.rand(1, *shape))
        x = self.conv(x)
        return int(np.prod(x.size()))

    def forward(
        self, audio, video, batch_size, time_step, hidden_state, cell_state
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        video = video.view(batch_size * time_step, *self.state_dim)
        video = video.float() / 255.0
        video_feature = self.video_feature(video)
        audio_feature = self.audio_feature(audio)
        x = torch.cat((audio_feature, video_feature), dim=1)
        x = self.feature(x)
        x = x.view(batch_size, time_step, -1)
        x, (hidden_state, cell_state) = self.lstm(x, (hidden_state, cell_state))
        x = x[:, -1, :]
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(), (hidden_state, cell_state)

    def init_hidden_states(self, bsize):
        h = torch.zeros(1, bsize, self.hidden_dim).float().to(self.device)
        c = torch.zeros(1, bsize, self.hidden_dim).float().to(self.device)
        return h, c


class DuelingBimodalDRQNAgent(Agent):
    def __init__(
        self,
        state_dim,
        sound_dim,
        action_dim,
        hidden_dim,
        kernel_size,
        stride,
        buffer_size,
        batch_size,
        time_step,
        gamma,
        learning_rate,
        weight_decay,
        device,
        model_class=DuelingBimodalDRQN,
        stack_size=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.time_step = time_step
        self.kernel_size = kernel_size
        self.stride = stride
        self.device = device
        self.sound_dim = sound_dim
        self.policy_net = model_class(
            state_dim,
            sound_dim,
            action_dim,
            kernel_size,
            stride,
            hidden_dim,
            device=device,
        ).to(device)
        self.target_net = model_class(
            state_dim,
            sound_dim,
            action_dim,
            kernel_size,
            stride,
            hidden_dim,
            device=device,
        ).to(device)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.buffer_size = buffer_size
        self.replay_buffer = RecurrentReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    @property
    def config(self):
        return {
            "architecture": "Dueling bimodal DRQN",
            "state_dim": self.state_dim,
            "sound_dim": self.sound_dim,
            "action_dim": self.action_dim,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "hidden_dim": self.hidden_dim,
            "buffer_size": self.buffer_size,
            "time_step": self.time_step,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
            "loss_fn": self.loss_fn,
            "weight_decay": self.weight_decay,
        }

    def load(self, path):
        pass

    def select_action(self, state, testing, **kwargs):
        epsilon = kwargs["epsilon"]
        hidden_state = kwargs["hidden_state"]
        cell_state = kwargs["cell_state"]
        audio_state = state["sound"]
        video_state = state["vision"]
        audio_state = torch.FloatTensor(audio_state).unsqueeze(0).to(self.device)
        video_state = torch.FloatTensor(video_state).unsqueeze(0).to(self.device)
        self.policy_net.eval()  # TODO: check if this is correct
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
        if np.random.rand() <= epsilon and not testing:
            # print("random action")
            action = np.random.choice(self.action_dim)
            return action, (new_hidden_state, new_cell_state)
        else:
            # print("policy action")
            action = current_qs.argmax().item()
            return action, (new_hidden_state, new_cell_state)

    def update_model(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return

        hidden_batch, cell_batch = self.policy_net.init_hidden_states(
            bsize=self.batch_size
        )
        batch_episodes: List[BimodalEpisode] = self.replay_buffer.get_batch(
            self.batch_size, self.time_step
        )

        audios_batch = []
        videos_batch = []
        actions_batch = []
        next_audios_batch = []
        next_videos_batch = []
        rewards_batch = []
        done_batch = []
        for episode in batch_episodes:
            (
                episode_audios,
                episode_videos,
                episode_actions,
                episode_next_audios,
                episode_next_videos,
                episode_rewards,
                episode_done,
            ) = zip(*episode)
            audios_batch.append(np.asarray(episode_audios))
            videos_batch.append(np.asarray(episode_videos))
            actions_batch.append(np.asarray(episode_actions))
            next_audios_batch.append(np.asarray(episode_next_audios))
            next_videos_batch.append(np.asarray(episode_next_videos))
            rewards_batch.append(np.asarray(episode_rewards))
            done_batch.append(np.asarray(episode_done))

        audios_batch_np = np.stack(audios_batch)
        videos_batch_np = np.stack(videos_batch)
        actions_batch_np = np.stack(actions_batch)
        next_audios_batch_np = np.stack(next_audios_batch)
        next_videos_batch_np = np.stack(next_videos_batch)
        rewards_batch_np = np.stack(rewards_batch)
        done_batch_np = np.stack(done_batch)

        torch_audios_batch = torch.FloatTensor(audios_batch_np).to(self.device)
        torch_videos_batch = torch.FloatTensor(videos_batch_np).to(self.device)
        torch_actions_batch = torch.FloatTensor(actions_batch_np).to(self.device)
        torch_next_audios_batch = torch.FloatTensor(next_audios_batch_np).to(
            self.device
        )
        torch_next_videos_batch = torch.FloatTensor(next_videos_batch_np).to(
            self.device
        )
        torch_rewards_batch = torch.FloatTensor(rewards_batch_np).to(self.device)
        torch_done_batch = torch.FloatTensor(done_batch_np).to(self.device)

        q_values, _ = self.policy_net.forward(
            torch_audios_batch,
            torch_videos_batch,
            self.batch_size,
            self.time_step,
            hidden_batch,
            cell_batch,
        )

        next_q_values, _ = self.target_net.forward(
            torch_next_audios_batch,
            torch_next_videos_batch,
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

    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def add_episode(self, episode: Episode):
        self.replay_buffer.add_episode(episode)
