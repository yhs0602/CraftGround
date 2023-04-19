from collections import deque

import numpy as np

import mydojo
from dqn import DQNAgent

if __name__ == "__main__":
    env = mydojo.make(
        initialInventoryCommands=["minecraft:diamond_sword", "minecraft:shield"],
        initialPosition=None,  # nullable
        initialMobsCommands=["minecraft:sheep"],
        imageSizeX=400,
        imageSizeY=225,
        seed=123456,  # nullable
        allowMobSpawn=True,
        alwaysDay=False,
        alwaysNight=False,
        initialWeather="clear",  # nullable
    )

    state_dim = env.observation_space.shape
    action_dim = env.action_space.n

    print(f"{state_dim=} {action_dim=}")

    hidden_dim = 32
    buffer_size = 1000000
    batch_size = 20
    gamma = 0.95
    learning_rate = 0.001

    agent = DQNAgent(
        state_dim, action_dim, buffer_size, batch_size, gamma, learning_rate
    )

    num_episodes = 1000
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    max_steps_per_episode = 100
    update_frequency = 50  # update target network every 100 steps

    scores = deque(maxlen=400)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps_per_episode):
            action = agent.select_action(state, epsilon)
            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            agent.add_experience(state, action, next_state, reward, done)
            agent.update_model()

            if step % update_frequency == 0:
                agent.update_target_model()  # important!

            if done:
                break

            state = next_state

        scores.append(episode_reward)
        print(
            f"Episode {episode}: score={episode_reward:.2f}, avg_score={np.mean(scores):.2f}, eps={epsilon:.2f}"
        )
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        if np.mean(scores) >= 195.0 and episode >= 100:
            print(f"Solved in {episode} episodes!")
            break
    env.render()
    env.close()
