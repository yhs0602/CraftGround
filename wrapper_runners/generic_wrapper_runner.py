import abc
import glob
import os
import time
from collections import deque
from typing import Optional

import numpy as np
import wandb
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

from mydojo.MyEnv import print_with_time
from dotenv import load_dotenv


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

    def update_model(self):
        pass


class GenericWrapperRunner:
    def __init__(
        self,
        env,
        env_name,
        group,
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
            "max_steps_per_episode": max_steps_per_episode,
            "num_episodes": num_episodes,
            "test_frequency": test_frequency,
        }

        config.update(agent.config)
        config.update(extra_configs)
        load_dotenv()
        wandb.init(
            # set the wandb project where this run will be logged
            project="mydojo",
            entity="jourhyang123",
            # track hyperparameters and run metadata
            config=config,
            resume=resume,
            group=group,
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
        recent_test_scores = deque(maxlen=10)
        scores = []
        avg_scores = []
        avg_test_scores = []
        avg_score = None
        avg_test_score = None
        test_score = None
        accum_steps = 0
        for episode in range(initial_episode, self.num_episodes):
            testing = episode % self.test_frequency == 0
            self.before_episode(episode, testing)
            if testing:
                test_score, num_steps, time_took, video_recorder = self.test_agent(
                    episode, record_video
                )
                recent_test_scores.append(test_score)
                avg_test_score = np.mean(recent_test_scores)
                avg_test_scores.append(avg_test_score)
            else:
                (
                    episode_reward,
                    num_steps,
                    accum_steps,
                    time_took,
                    avg_loss,
                    reset_extra_info,
                ) = self.train_agent(episode, accum_steps)
                if num_steps == 0:
                    num_steps = 1
                scores.append(episode_reward)
                recent_scores.append(episode_reward)
                avg_score = np.mean(recent_scores)
                avg_scores.append(avg_score)
                thing_to_log = {
                    "episode": episode,
                    "score": episode_reward,
                    "avg_score": avg_score,
                    "avg_loss": avg_loss,
                }
                thing_to_log.update(self.get_extra_log_info())
                # thing_to_log.update(reset_extra_info)
                print(
                    " ".join(["{0}={1}".format(k, v) for k, v in thing_to_log.items()])
                )
                wandb.log(thing_to_log)

            print(
                f"Seconds per episode{episode}: {time_took}/{num_steps}={time_took / num_steps:.5f} seconds"
            )

            self.after_episode(episode, testing)
            if self.solved_criterion(avg_score, test_score, avg_test_score, episode):
                print(f"Solved in {episode} episodes!")
                break

        self.env.terminate()
        self.env.close()
        wandb.finish()

    def train_agent(self, episode, accum_steps):
        state, reset_info = self.env.reset(fast_reset=True)
        # reset_extra_info = reset_info.get("extra_info", None)
        print_with_time("Finished resetting the environment")
        episode_reward = 0
        sum_time = 0
        num_steps = 0
        losses = []
        for step in range(self.max_steps_per_episode):
            start_time = time.time()
            action = self.select_action(episode, state, False)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            episode_reward += reward
            accum_steps += 1
            num_steps += 1
            loss = self.after_step(
                step,
                accum_steps,
                state,
                action,
                next_state,
                reward,
                terminated,
                truncated,
                info,
                False,
            )
            losses.append(loss)
            if terminated:
                break

            state = next_state
            elapsed_time = time.time() - start_time
            # print(f"Step {step} took {elapsed_time:.5f} seconds")
            sum_time += elapsed_time
        avg_loss = np.mean([loss for loss in losses if loss is not None])
        return (
            episode_reward,
            num_steps,
            accum_steps,
            sum_time,
            avg_loss,
            None,
        )  # reset_extra_info

    def test_agent(self, episode, record_video):
        if record_video:
            video_recorder = VideoRecorder(self.env, f"video{episode}.mp4")
        state, info = self.env.reset(fast_reset=True)
        # reset_extra_info = state.get("extra_info", None)
        print_with_time("Finished resetting the environment")
        episode_reward = 0
        time_took = 0
        num_steps = 0
        for step in range(self.max_steps_per_episode):
            start_time = time.time()
            if record_video:
                video_recorder.capture_frame()
            action = self.select_action(episode, state, True)
            # print(f"testing {action=} state norm = {np.linalg.norm(state)}")
            # print(f"{state=}, {action=}")
            next_state, reward, terminated, truncated, info = self.env.step(action)
            # extra_info = info.get("extra_info", None)
            episode_reward += reward
            self.after_step(
                step,
                step,
                state,
                action,
                next_state,
                reward,
                terminated,
                truncated,
                info,
                True,
            )

            if terminated:
                break
            state = next_state
            elapsed_time = time.time() - start_time
            # print(f"Step {step} took {elapsed_time:.5f} seconds")
            time_took += elapsed_time
            num_steps += 1
        if record_video:
            video_recorder.close()
        to_log = {"test/score": episode_reward, "test/step": episode}
        # to_log.update(reset_extra_info)
        wandb.log(to_log)
        return episode_reward, num_steps, time_took, video_recorder

    def before_training(self):
        pass

    def before_episode(self, episode, testing: bool):
        pass

    def select_action(self, episode, state, testing):
        return self.agent.select_action(state, testing)

    def after_step(
        self,
        step,
        accum_steps,
        state,
        action,
        next_state,
        reward,
        terminated,
        truncated,
        info,
        testing: bool,
    ) -> Optional[float]:
        pass

    def after_episode(self, episode, testing: bool):
        pass

    def get_extra_log_info(self):
        return {}
