import glob
from collections import deque

import numpy as np

import mydojo
from models.dqn import DQNAgent
import os

max_saved_models = 3
model_dir = "saved_models"
plot_filename = "figures/scores.png"

import matplotlib.pyplot as plt


def save_score_plot(scores, avg_scores, filename):
    # Remove the file if it already exists
    if os.path.isfile(filename):
        os.remove(filename)

    # Create the plot
    x = np.arange(len(scores))
    fig, ax = plt.subplots()
    ax.plot(x, scores, label="Score")
    ax.plot(x, avg_scores, label="Avg Score")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.legend()

    # Save the plot to a file
    plt.savefig(filename)


def load_latest_model(agent, directory):
    # Find the latest saved model file
    list_of_files = glob.glob(os.path.join(directory, "*.pt"))
    if len(list_of_files) == 0:
        return 0, 1.0  # No saved models

    latest_file = max(list_of_files, key=os.path.getctime)

    episode_number = int(latest_file.split("episode_")[1].split(".")[0])
    # Load the model from the file
    epsilon_ = agent.load(latest_file)
    return episode_number, epsilon_


if __name__ == "__main__":
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    env = mydojo.make(
        initialInventoryCommands=["minecraft:diamond_sword", "minecraft:shield"],
        initialPosition=None,  # nullable
        initialMobsCommands=[
            "minecraft:sheep",
            "minecraft:husk ~10 ~ ~ {HandItems:[{Count:1,id:wooden_shovel},{}]}",
        ],
        imageSizeX=64,
        imageSizeY=64,
        visibleSizeX=400,
        visibleSizeY=225,
        seed=12345,  # nullable
        allowMobSpawn=False,
        alwaysDay=False,
        alwaysNight=False,
        initialWeather="clear",  # nullable
        isHardCore=False,
        isWorldFlat=True,  # superflat world
    )

    state_dim = env.observation_space.shape
    action_dim = env.action_space.n

    # print(f"{state_dim=} {action_dim=}")

    hidden_dim = 32
    buffer_size = 1000000
    batch_size = 20
    gamma = 0.95
    learning_rate = 0.001

    agent = DQNAgent(
        state_dim, action_dim, buffer_size, batch_size, gamma, learning_rate
    )

    fresh_run = True

    initial_epsiode = 0
    epsilon = 1.0

    if not fresh_run:
        initial_epsiode, epsilon = load_latest_model(agent, model_dir)

    num_episodes = 1000
    epsilon_min = 0.01
    epsilon_decay = 0.995
    max_steps_per_episode = 400
    update_frequency = 50  # update target network every 100 steps

    recent_scores = deque(maxlen=30)
    scores = []
    avg_scores = []

    for episode in range(initial_epsiode, num_episodes):
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

        # Save the agent's model
        model_path = os.path.join(model_dir, f"model_episode_{episode}.pt")
        agent.save(model_path, epsilon)
        # Delete oldest saved model if there are more than max_saved_models
        saved_models = sorted(
            os.listdir(model_dir),
            key=lambda x: os.path.getctime(os.path.join(model_dir, x)),
        )
        while len(saved_models) > max_saved_models:
            oldest_model_path = os.path.join(model_dir, saved_models[0])
            os.remove(oldest_model_path)
            saved_models.pop(0)

        scores.append(episode_reward)
        recent_scores.append(episode_reward)
        avg_score = np.mean(recent_scores)
        avg_scores.append(avg_score)
        print(
            f"Episode {episode}: score={episode_reward:.2f}, avg_score={avg_score:.2f}, eps={epsilon:.2f}"
        )
        save_score_plot(scores, avg_scores, plot_filename)
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        if avg_score >= 300.0 and episode >= 100:
            print(f"Solved in {episode} episodes!")
            break
    env.close()
