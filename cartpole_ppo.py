from collections import deque

import gym
import numpy as np
import torch

from models.ppo import PPO


def main():
    ################################## set device ##################################
    print(
        "============================================================================================"
    )
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
    elif torch.has_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device set to : " + device.type)
    # Create the environment
    env = gym.make("CartPole-v1", render_mode="human")

    max_ep_len = 1000
    update_timestep = max_ep_len * 4
    max_training_timesteps = int(3e6)
    rewards = deque(maxlen=100)

    # Create and train the PPO agent
    agent = PPO(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,  # 2
        lr_actor=0.0001,
        lr_critic=0.0001,
        gamma=0.99,
        K_epochs=4,
        eps_clip=0.2,
        has_continuous_action_space=False,
        device=device,
        action_std_init=0.6,
    )
    # train the agent
    time_step = 0
    episode = 0
    while time_step < max_training_timesteps:
        state, info = env.reset()
        done = False

        total_reward = 0
        current_time_step = 0
        while not done and current_time_step < max_ep_len:
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
            time_step += 1
            current_time_step += 1
            total_reward += reward
            if time_step % update_timestep == 0:
                print("Updating")
                agent.update()
            # if episode % 10 == 0:
            #     print(f"render {episode=}")
            #     env.render()
            state = next_state
        episode += 1
        rewards.append(total_reward)
        if episode % 10 == 0:
            print(
                f"Episode {episode + 1} completed with reward {total_reward}, average reward {np.mean(rewards)}"
            )

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


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main()
