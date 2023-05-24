import abc
import glob
import os
import time
from collections import deque

import numpy as np
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

import wandb
from mydojo.MyEnv import print_with_time


class Agent(abc.ABC):
    @property
    @abc.abstractmethod
    def config(self):
        pass

    @abc.abstractmethod
    def select_action(self, obs, testing, **kwargs):
        pass

    @abc.abstractmethod
    def load(self, path):
        pass


def x(warmup_episodes,
      update_frequency,
      epsilon_init,
      epsilon_min,
      epsilon_decay):
    dqn_config = {
        "warmup_episodes": warmup_episodes,
        "update_frequency": update_frequency,
        "epsilon_init": epsilon_init,
        "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay,
    }


class GenericWrapperRunner:
    def __init__(
            self,
            env,
            env_name,
            agent: Agent,
            max_steps_per_episode,
            num_episodes,
            test_frequency,
            solved_criterion,
            after_wandb_init: callable,
            resume: bool = False,
            max_saved_models=2,
            **extra_configs,
    ):
        config = {
            "environment": env_name,
            "architecture": "DQNAgent",
            "max_steps_per_episode": max_steps_per_episode,
            "num_episodes": num_episodes,
            "test_frequency": test_frequency,
        }

        config.update(agent.config)
        config.update(extra_configs)
        wandb.init(
            # set the wandb project where this run will be logged
            project="mydojo",
            entity="jourhyang123",
            # track hyperparameters and run metadata
            config=config,
            resume=resume,
        )
        # define our custom x axis metric
        wandb.define_metric("test/step")
        # define which metrics will be plotted against it
        wandb.define_metric("test/*", step_metric="test/step")
        after_wandb_init()

        self.env = env
        self.agent = agent
        self.max_steps_per_episode = max_steps_per_episode
        self.num_episodes = num_episodes
        self.test_frequency = test_frequency
        self.solved_criterion = solved_criterion
        self.model_dir = os.path.join(wandb.run.dir, env_name)
        self.local_plot_filename = os.path.join(self.model_dir, f"{env_name}.png")
        self.max_saved_models = max_saved_models
        self.extra_configs = extra_configs
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def load_latest_model(self, agent):
        # Find the latest saved model file
        list_of_files = glob.glob(os.path.join(self.model_dir, "*.pt"))
        if len(list_of_files) == 0:
            return 0, 1.0  # No saved models

        latest_file = max(list_of_files, key=os.path.getctime)

        episode_number = int(latest_file.split("episode_")[1].split(".")[0])
        # Load the model from the file
        epsilon_ = agent.load(latest_file)
        return episode_number, epsilon_

    def save_model(self, episode: int, agent: Agent, epsilon: float):
        model_path = os.path.join(self.model_dir, f"model_episode_{episode}.pt")
        agent.save(model_path, epsilon)
        # Delete oldest saved model if there are more than max_saved_models
        saved_models = sorted(
            os.listdir(self.model_dir),
            key=lambda x: os.path.getctime(os.path.join(self.model_dir, x)),
        )
        while len(saved_models) > self.max_saved_models:
            oldest_model_path = os.path.join(self.model_dir, saved_models[0])
            os.remove(oldest_model_path)
            saved_models.pop(0)

    def run_wrapper(self, record_video=False):
        if record_video:
            wandb.gym.monitor()
        initial_episode = 0
        self.before_training()

        recent_scores = deque(maxlen=30)
        scores = []
        avg_scores = []
        for episode in range(initial_episode, self.num_episodes):
            self.before_episode(episode)
            testing = episode % self.test_frequency == 0
            if testing and record_video:
                video_recorder = VideoRecorder(self.env, f"video{episode}.mp4")

            state = self.env.reset(fast_reset=True)
            print_with_time("Finished resetting the environment")
            episode_reward = 0
            sum_time = 0
            num_steps = 0
            for step in range(self.max_steps_per_episode):
                start_time = time.time()
                if testing and record_video:
                    video_recorder.capture_frame()

                action = self.select_action(episode, state, testing)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                self.after_step(step, state, action, next_state, reward, terminated, truncated, info)

                if terminated:
                    break

                state = next_state
                elapsed_time = time.time() - start_time
                # print(f"Step {step} took {elapsed_time:.5f} seconds")
                sum_time += elapsed_time
                num_steps += 1

            if num_steps == 0:
                num_steps = 1
            print(
                f"Seconds per episode{episode}: {sum_time}/{num_steps}={sum_time / num_steps:.5f} seconds"
            )

            scores.append(episode_reward)
            recent_scores.append(episode_reward)
            avg_score = np.mean(recent_scores)
            avg_scores.append(avg_score)
            thing_to_log = {
                "episode": episode,
                "score": episode_reward,
                "avg_score": avg_score,
            }
            thing_to_log.update(self.get_extra_log_info())
            print(' '.join(['{0}={1}'.format(k, v) for k, v in thing_to_log.items()]))
            if testing:
                video_recorder.close()
                wandb.log({"test/score": episode_reward, "test/step": episode})
            else:
                wandb.log(thing_to_log)

            self.after_episode(episode)

            if self.solved_criterion(avg_score, episode):
                print(f"Solved in {episode} episodes!")
                break

        self.env.close()
        wandb.finish()

    def before_training(self):
        pass

    def before_episode(self, episode):
        pass

    def select_action(self, episode, state, testing):
        return self.agent.select_action(state, testing)

    def after_step(self, step, state, action, next_state, reward, terminated, truncated, info):
        pass

    def after_episode(self, avg_score):
        pass

    def get_extra_log_info(self):
        return {}


class DQNWrapperRunner(GenericWrapperRunner):
    def __init__(
            self,
            env,
            env_name,
            agent: 'DQNAgent',
            max_steps_per_episode,
            num_episodes,
            test_frequency,
            solved_criterion,
            after_wandb_init: callable,
            warmup_episodes,
            update_frequency,
            epsilon_init,
            epsilon_min,
            epsilon_decay,
            resume: bool = False,
            max_saved_models=2,
    ):
        import models.dqn
        def after_wandb_init_fn():
            models.dqn.after_wandb_init()
            after_wandb_init()

        super().__init__(
            env,
            env_name,
            agent,
            max_steps_per_episode,
            num_episodes,
            test_frequency,
            solved_criterion,
            after_wandb_init_fn,
            resume,
            max_saved_models,
            warmup_episodes=warmup_episodes,
            update_frequency=update_frequency,
            epsilon_init=epsilon_init,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
        )
        self.epsilon = epsilon_init
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.warmup_episodes = warmup_episodes
        self.update_frequency = update_frequency

    def select_action(self, episode, state, testing):
        if episode < self.warmup_episodes:
            action = self.env.action_space.sample()
        else:
            action = self.agent.select_action(state, testing, epsilon=self.epsilon)
        return action

    def after_step(self, step, state, action, next_state, reward, terminated, truncated, info):
        self.agent.add_experience(state, action, next_state, reward, terminated)
        self.agent.update_model()
        if step % self.update_frequency == 0:
            self.agent.update_target_model()  # important! FIXME: step ranges from 0 to max_steps_per_episode;

    def after_episode(self, episode):
        if episode >= self.warmup_episodes:
            self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

    def get_extra_log_info(self):
        return {
            "epsilon": self.epsilon
        }


def main():
    from wrappers.EscapeHuskBySoundWithPlayerWrapper import EscapeHuskBySoundWithPlayerWrapper
    env = EscapeHuskBySoundWithPlayerWrapper()
    buffer_size = 1000000
    batch_size = 256
    gamma = 0.99
    learning_rate = 0.0001  # 0.001은 너무 크다
    update_freq = 1000  # 에피소드 여러 개 하면서 학습하게 1000 이렇게 하고 줄이기
    hidden_dim = 128  # 128정도 해보기
    # weight_decay = 0.0001
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    from models.dqn import DQNSoundAgent
    agent = DQNSoundAgent(
        state_dim,
        action_dim,
        hidden_dim,
        buffer_size,
        batch_size,
        gamma,
        learning_rate,
        # weight_decay=weight_decay,
    )
    runner = DQNWrapperRunner(
        env,
        env_name="HuskSound-6-6-yaw",
        agent=agent,
        max_steps_per_episode=400,
        num_episodes=700,
        test_frequency=10,
        solved_criterion=lambda avg_score, episode: avg_score >= 190.0
                                                    and episode >= 300,
        after_wandb_init=lambda *args: None,
        warmup_episodes=0,
        update_frequency=update_freq,
        epsilon_init=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99,
        resume=False,
        max_saved_models=2,
    )
    runner.run_wrapper(record_video=True)


if __name__ == "__main__":
    main()
