import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DuelingBiModalAttentionDQN(nn.Module):
    def __init__(
        self,
        state_dim,
        sound_dim,
        action_dim,
        kernel_size,
        stride,
        hidden_dim,
        attention_dim,
    ):
        super(DuelingBiModalAttentionDQN, self).__init__()
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
        conv_out_size = self._get_conv_output(state_dim)
        self.video_feature = nn.Sequential(
            self.conv, nn.Flatten(), nn.Linear(conv_out_size, hidden_dim), nn.ReLU()
        )
        self.audio_attention = nn.Linear(hidden_dim, attention_dim)
        self.video_attention = nn.Linear(hidden_dim, attention_dim)
        self.attention_combine = nn.Linear(attention_dim, 1)
        self.feature = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def _get_conv_output(self, shape):
        x = Variable(torch.rand(1, *shape))
        x = self.conv(x)
        return int(np.prod(x.size()))

    def _apply_attention(self, feature, attention):
        weights = F.softmax(attention(feature), dim=1)
        weighted_feature = weights * feature
        attended_feature = torch.sum(weighted_feature, dim=1)
        return attended_feature

    def forward(self, audio, video):
        video = video.float() / 255.0
        audio_feature = self.audio_feature(audio)
        video_feature = self.video_feature(video)

        audio_attended = self._apply_attention(audio_feature, self.audio_attention)
        video_attended = self._apply_attention(video_feature, self.video_attention)

        print(f"{audio.shape=} {audio_feature.shape=} {audio_attended.shape=}")
        print(f"{video.shape=} {video_feature.shape=} {video_attended.shape=}")

        x = torch.cat((audio_attended, video_attended), dim=1)
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
