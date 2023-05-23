import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Define the PPO policy network
class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x


# Define the PPO agent
class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.policy = Policy(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.001)
        self.gamma = 0.99
        self.epsilon = 0.2

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probabilities = self.policy(state)
        action = torch.multinomial(probabilities, 1).item()
        return action

    def compute_returns(self, rewards, masks):
        returns = []
        R = 0
        for reward, mask in zip(reversed(rewards), reversed(masks)):
            R = reward + self.gamma * R * mask
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        return returns

    def update_policy(self, states, actions, returns, advantages):
        old_probabilities = self.policy(states).detach().clone()
        old_probabilities = old_probabilities.gather(1, actions.unsqueeze(1))

        for x in range(10):  # PPO optimization epochs
            print(f"{x=}")
            probabilities = self.policy(states)
            ratios = probabilities / old_probabilities
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            loss = -torch.min(surr1, surr2).mean()

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state, info = self.env.reset()
            done = False

            states = []
            actions = []
            rewards = []
            masks = []
            values = []
            advantages = []

            while not done:
                action = self.select_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                masks.append(1 - done)

                state = next_state

            returns = self.compute_returns(rewards, masks)

            for t in range(len(states)):
                state = torch.from_numpy(states[t]).float()
                action = torch.tensor(actions[t]).unsqueeze(0)
                value = self.policy(state.unsqueeze(0)).squeeze()

                advantages.append(returns[t] - value)
                values.append(value)

            advantages = torch.stack(advantages)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            states = torch.from_numpy(np.stack(states)).float()
            actions = torch.tensor(actions, dtype=torch.int64)
            returns = torch.from_numpy(np.stack(returns)).float()

            self.update_policy(states, actions, returns, advantages)

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes} completed.")

        print("Training completed.")


def main():
    # Create the environment
    env = gym.make('CartPole-v1')

    # Create and train the PPO agent
    agent = PPOAgent(env)
    agent.train(num_episodes=100)

    # Evaluate the trained agent
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        env.render()

    print(f"Total reward: {total_reward}")

    # Close the environment
    env.close()


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main()
