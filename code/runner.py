import argparse
import time

import numpy as np
import yaml

import algorithm
import wrappers
from environments import env_makers
from get_device import get_device
from logger import Logger


class Runner:
    def __init__(self, config_filename: str, verbose: bool = False):
        with open(config_filename, "r") as yaml_file:
            yaml_content = yaml_file.read()
        data = yaml.safe_load(yaml_content)
        self.verbose = verbose
        self.data = data
        self.seed = data["seed"]
        self.env_path = data["env_path"]
        self.group = data["group"]
        self.record_video = data["record_video"]
        self.device = data["device"]
        if self.device is None:
            self.device = get_device()

        env_data = data["env"]
        self.env_name = env_data["name"]
        self.env_params = env_data["params"]
        self.wrappers = data["wrappers"]
        self.algorithm_data = data["algorithm"]
        self.algorithm_name = self.algorithm_data["name"]
        self.algorithm_params = self.algorithm_data["params"]

    def run(self):
        if self.seed is None:
            self.seed = int(time.time())
        np.random.seed(self.seed)
        env_params = self.env_params
        if self.verbose:
            env_params["verbose"] = True
        inner_env, sound_list = env_makers[self.env_name](
            env_path=self.env_path, **self.env_params
        )
        for wrapper in self.wrappers:
            wrapper_class = getattr(wrappers, wrapper["name"])
            wrapper_instance = wrapper_class(
                env=inner_env, sound_list=sound_list, **wrapper
            )
            inner_env = wrapper_instance

        env = inner_env

        self.data.update({"seed": self.seed})
        self.data.update({"sound_list": sound_list})
        logger = Logger(
            group=self.group,
            resume=False,
            config=self.data,
            record_video=self.record_video,
        )

        # RUN ALGORITHM
        # algorithm = DQN | DRQN | A2C | PPO | SAC ...
        algorithm_class = getattr(algorithm, self.algorithm_name)
        algorithm_instance = algorithm_class(
            env=env, **self.algorithm_params, logger=logger, device=self.device
        )
        algorithm_instance.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python runner.py",
        description="Run an experiment.",
        epilog="Example: python runner.py --config experiments/husk.yaml",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the config file.",
        required=True,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print the log of gradle",
        default=False,
        required=False,
    )
    args = parser.parse_args()
    runner = Runner(config_filename=args.config, verbose=args.verbose)
    runner.run()
