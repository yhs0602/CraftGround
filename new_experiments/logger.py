from typing import Dict

import wandb
from dotenv import load_dotenv
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder


class Logger:
    def __init__(self, config, resume, group, record_video):
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
        self.record_video = record_video
        self.video_recorder = None

    def log(self, data: Dict):
        print(" ".join(["{0}={1}".format(k, v) for k, v in data.items()]))
        wandb.log(data)

    def start_training(self):
        if self.record_video:
            wandb.gym.monitor()

    def before_episode(self, env, should_record_video: bool, episode: int):
        if self.record_video and should_record_video:
            self.video_recorder = VideoRecorder(env, f"video{episode}.mp4")

    def before_step(self, step, should_record_video: bool):
        if self.record_video and should_record_video and self.video_recorder:
            self.video_recorder.capture_frame()

    def after_episode(self):
        if self.record_video and self.video_recorder:
            self.video_recorder.close()
            self.video_recorder = None
