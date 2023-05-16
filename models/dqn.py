import random
from collections import deque, namedtuple

import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.has_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# Define the DQN class with a CNN architecture
class DQN(nn.Module):
    def __init__(
        self, input_shape, num_actions, kernel_size=5, stride=2, hidden_dim=512
    ):
        super(DQN, self).__init__()
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


Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def add(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        state = torch.stack(list(map(torch.from_numpy, batch.state)))
        action = torch.stack([torch.Tensor([float(x)]) for x in batch.action])
        reward = torch.stack([torch.Tensor([float(x)]) for x in batch.reward])
        next_state = torch.stack(list(map(torch.from_numpy, batch.next_state)))
        done = torch.stack([torch.Tensor([x]) for x in batch.done])
        return state, action, next_state, reward, done  # tuple(map(torch.cat, batch))


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        buffer_size=100000,
        batch_size=32,
        gamma=0.99,
        learning_rate=0.001,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

    def select_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            # print("random action")
            return np.random.choice(self.action_dim)
        else:
            # print("policy action")
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.argmax().item()

    def update_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        # print("Will update model")
        state, action, next_state, reward, done = self.replay_buffer.sample(
            self.batch_size
        )
        state = state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        next_state = next_state.to(device)
        done = done.to(device)

        q_values = self.policy_net(state).gather(1, action.to(torch.int64)).squeeze(1)
        next_q_values = self.target_net(next_state).max(1)[0]
        expected_q_values = reward + (1 - done) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, expected_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def add_experience(self, state, action, next_state, reward, done):
        self.replay_buffer.add(state, action, next_state, reward, done)

    def save(self, path, epsilon):
        state_dict = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "epsilon": epsilon,
        }
        torch.save(state_dict, path)

    def load(self, path):
        state_dict = torch.load(path)
        self.policy_net.load_state_dict(state_dict["policy_net"])
        self.target_net.load_state_dict(state_dict["target_net"])
        return state_dict.get("epsilon", 1)

    @property
    def config(self):
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "buffer_size": self.buffer_size,
            "learning_rage": self.learning_rate,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
            "loss_fn": self.loss_fn,
        }


class SoundDQN(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_dim=32):
        super(SoundDQN, self).__init__()
        self.fc1 = nn.Linear(input_shape[0], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQNSoundAgent(DQNAgent):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=32,
        buffer_size=100000,
        batch_size=32,
        gamma=0.99,
        learning_rate=0.001,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.policy_net = SoundDQN(state_dim, action_dim, hidden_dim).to(device)
        self.target_net = SoundDQN(state_dim, action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

    @property
    def config(self):
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "buffer_size": self.buffer_size,
            "learning_rage": self.learning_rate,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
            "loss_fn": self.loss_fn,
        }
