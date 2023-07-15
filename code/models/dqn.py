import numpy as np
import torch
import wandb
from torch import nn
from torch.autograd import Variable

from get_device import get_device

device = get_device()


def after_wandb_init():
    wandb.run.log_code(".")


class DQNSoundAndVision(nn.Module):
    def __init__(
        self, input_shape, sound_dim, num_actions, kernel_size, stride, hidden_dim
    ):
        super(DQNSoundAndVision, self).__init__()
        self.conv1 = nn.Conv2d(
            input_shape[0], 16, kernel_size=kernel_size, stride=stride
        )  # (210, 160, 3), permuted to (3, 210, 160)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride)
        self.cnn_fc1 = nn.Linear(self.get_conv_output(input_shape), hidden_dim)
        self.sound_fc1 = nn.Linear(sound_dim[0], hidden_dim)
        self.sound_bn1 = nn.BatchNorm1d(hidden_dim)
        self.sound_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.sound_bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim * 2, num_actions)

    def forward(self, x, sound):
        x = x.float() / 255.0
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.cnn_fc1(x))
        s_x = nn.functional.relu(self.sound_bn1(self.sound_fc1(sound)))
        s_x = nn.functional.relu(self.sound_bn2(self.sound_fc2(s_x)))
        x = torch.cat((x, s_x), dim=1)
        x = self.fc3(x)
        return x

    def get_conv_output(self, shape):
        x = Variable(torch.rand(1, *shape))
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        return int(np.prod(x.size()))


# Define the DQN class with a CNN architecture
class CNNDQN(nn.Module):
    def __init__(self, input_shape, num_actions, kernel_size, stride, hidden_dim):
        super(CNNDQN, self).__init__()
        self.conv1 = nn.Conv2d(
            input_shape[0], 16, kernel_size=kernel_size, stride=stride
        )  # (210, 160, 3), permuted to (3, 210, 160)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride)
        self.fc1 = nn.Linear(self.get_conv_output(input_shape), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = x.float() / 255.0
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_conv_output(self, shape):
        x = Variable(torch.rand(1, *shape))
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        return int(np.prod(x.size()))


class CNNDQNWithBNPool(nn.Module):
    def __init__(self, input_shape, num_actions, kernel_size, stride, hidden_dim):
        super(CNNDQNWithBNPool, self).__init__()
        self.conv1 = nn.Conv2d(
            input_shape[0], 16, kernel_size=kernel_size, stride=stride
        )  # (210, 160, 3), permuted to (3, 210, 160)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(self.get_conv_output(input_shape), hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = x.float() / 255.0
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        return x

    def get_conv_output(self, shape):
        x = Variable(torch.rand(1, *shape))
        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool2(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.pool3(x)
        return int(np.prod(x.size()))


class SoundDQN(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_dim=128):
        super(SoundDQN, self).__init__()
        self.fc1 = nn.Linear(input_shape[0], hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        if x.dim() == 3:  # stacked frames
            x = x.view(-1, x.shape[1] * x.shape[2])
        x = nn.functional.relu(self.bn1(self.fc1(x)))
        x = nn.functional.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
