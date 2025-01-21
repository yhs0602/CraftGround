import os
import re
import signal
import socket
import struct
import subprocess
from enum import Enum
from typing import Tuple, Optional, Union, List, Any, Dict

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType, RenderFrame
import torch

from environment.action_space import (
    ActionSpaceVersion,
    declare_action_space,
    translate_action_to_v2,
)
from environment.observation_converter import ObservationConverter
from environment.observation_space import declare_observation_space
from environment.socket_ipc import SocketIPC

from buffered_socket import BufferedSocket
from csv_logger import CsvLogger, LogBackend
from initial_environment_config import InitialEnvironmentConfig
from proto import observation_space_pb2


class ObservationTensorType(Enum):
    NONE = 0
    CUDA_DLPACK = 1
    APPLE_TENSOR = 2


class CraftGroundEnvironment(gym.Env):
    def __init__(
        self,
        initial_env: InitialEnvironmentConfig,
        action_space_version: ActionSpaceVersion = ActionSpaceVersion.V1_MINEDOJO,
        verbose=False,
        env_path=None,
        port=8000,
        find_free_port: bool = True,
        use_shared_memory: bool = False,
        render_action: bool = False,
        render_alternating_eyes: bool = False,
        use_terminate: bool = False,
        cleanup_world: bool = True,  # removes the world when the environment is closed
        use_vglrun: bool = False,  # use vglrun to run the server (headless 3d acceleration)
        track_native_memory: bool = False,
        ld_preload: Optional[str] = None,
        native_debug: bool = False,
        verbose_python: bool = False,
        verbose_gradle: bool = False,
        verbose_jvm: bool = False,
        profile: bool = False,
    ):
        self.action_space_version = action_space_version
        self.action_space = declare_action_space(action_space_version)
        self.observation_space = declare_observation_space(
            initial_env.imageSizeX, initial_env.imageSizeY
        )
        self.initial_env = initial_env
        self.use_terminate = use_terminate
        self.cleanup_world = cleanup_world
        self.use_vglrun = use_vglrun
        self.native_debug = native_debug
        self.track_native_memory = track_native_memory
        self.ld_preload = ld_preload
        self.encoding_mode = initial_env.screen_encoding_mode
        self.sock = None
        self.buffered_socket = None
        self.last_rgb_frames: List[Union[np.ndarray, torch.Tensor, None]] = [None, None]
        self.last_images: List[Union[np.ndarray, torch.Tensor, None]] = [None, None]
        self.last_action = None
        self.render_action = render_action
        self.verbose = verbose
        self.verbose_python = verbose_python
        self.verbose_gradle = verbose_gradle
        self.verbose_jvm = verbose_jvm
        self.profile = profile

        self.render_alternating_eyes = render_alternating_eyes
        self.render_alternating_eyes_counter = 0
        self.find_free_port = find_free_port
        self.queued_commands = []
        self.process = None
        self.use_shared_memory = use_shared_memory
        if env_path is None:
            self.env_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "MinecraftEnv",
            )
        else:
            self.env_path = env_path
        self.csv_logger = CsvLogger(
            "py_log.csv",
            profile=profile,
            backend=LogBackend.BOTH if verbose_python else LogBackend.NONE,
        )

        if self.use_shared_memory:
            from .boost_ipc import BoostIPC  # type: ignore

            self.ipc = BoostIPC(str(port), initial_env)
        else:
            self.ipc = SocketIPC(self.csv_logger, port, find_free_port)

        # in case when using zerocopy
        self.observation_tensors = [None, None]
        self.observation_tensor_type = ObservationTensorType.NONE

        self.observation_converter = ObservationConverter(
            self.initial_env.screen_encoding_mode,
            self.initial_env.eye_distance > 0,
            self.render_action,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        if options is None:
            options = {}
        fast_reset = options.get("fast_reset", True)
        extra_commands = options.get("extra_commands", [])

        self.ipc.ensure_alive(fast_reset, extra_commands, seed=seed)

        with self.csv_logger.profile("read_response"):
            res = self.ipc.read_observation()
        final_obs = self.convert_observation_v2(res)
        return final_obs, final_obs

    def convert_observation_v2(self, res):
        with self.csv_logger.profile("convert_observation"):
            rgb_1, rgb_2 = self.observation_converter.convert(res)

        self.queued_commands = []
        res.yaw = ((res.yaw + 180) % 360) - 180
        final_obs: Dict[str, Union[np.ndarray, torch.Tensor, Any]] = {
            "full": res,
            "pov": rgb_1,
        }
        if rgb_2 is not None:
            final_obs["pov_2"] = rgb_2
        return final_obs

    def start_server(
        self,
        port: int,
        use_vglrun: bool,
        track_native_memory: bool,
        ld_preload: Optional[str],
    ):
        self.remove_orphan_java_processes()  # TODO
        # Check if a file exists

        # Prepare command TODO
        my_env = os.environ.copy()
        my_env["PORT"] = str(port)
        my_env["VERBOSE"] = str(int(self.verbose_jvm))
        if track_native_memory:
            my_env["CRAFTGROUND_JVM_NATIVE_TRACKING"] = "detail"
        if self.native_debug:
            my_env["CRAFGROUND_NATIVE_DEBUG"] = "True"
        # configure permission of the gradlew
        gradlew_path = os.path.join(self.env_path, "gradlew")
        if not os.access(gradlew_path, os.X_OK):
            os.chmod(gradlew_path, 0o755)
        # update image settings of options.txt if exists
        options_txt_path = self.get_env_option_path()
        if options_txt_path is not None:
            if os.path.exists(options_txt_path):
                pass
                # self.update_override_resolutions(options_txt_path)

        cmd = f"./gradlew runClient -w --no-daemon"  #  --args="--width {self.initial_env.imageSizeX} --height {self.initial_env.imageSizeY}"'
        if use_vglrun:
            cmd = f"vglrun {cmd}"
        if ld_preload:
            my_env["LD_PRELOAD"] = ld_preload
        self.csv_logger.log(f"Starting server with command: {cmd}")

        # Launch the server
        self.process = subprocess.Popen(
            cmd,
            cwd=self.env_path,
            shell=True,
            stdout=subprocess.DEVNULL if not self.verbose_gradle else None,
            env=my_env,
        )

        # TODO: socket specific
        sock: socket.socket = self.ipc.wait_for_server(port)
        self.sock = sock

        self.ipc.send_initial_environment(
            self.initial_env.to_initial_environment_message()
        )

        # TODO: socket specific
        self.buffered_socket = BufferedSocket(self.sock)
        self.csv_logger.log("Sent initial environment")

    def update_override_resolutions(self, options_txt_path):
        with open(options_txt_path, "r") as file:
            text = file.read()

            # Define the patterns for overrideWidth and overrideHeight
        width_pattern = r"overrideWidth:\d+"
        height_pattern = r"overrideHeight:\d+"

        # Update or add overrideWidth
        if re.search(width_pattern, text):
            text = re.sub(
                width_pattern, f"overrideWidth:{self.initial_env.imageSizeX}", text
            )
        else:
            text += f"\noverrideWidth:{self.initial_env.imageSizeX}"

        # Update or add overrideHeight
        if re.search(height_pattern, text):
            text = re.sub(
                height_pattern, f"overrideHeight:{self.initial_env.imageSizeY}", text
            )
        else:
            text += f"\noverrideHeight:{self.initial_env.imageSizeY}"

        # Write the updated text back to the file
        with open(options_txt_path, "w") as file:
            file.write(text)
        print(
            f"Updated {options_txt_path} to {self.initial_env.imageSizeX}x{self.initial_env.imageSizeY}"
        )

    def read_one_observation(self) -> Tuple[int, ObsType]:
        # print("Reading observation size...")
        data_len_bytes = self.buffered_socket.read(4, True)
        # print("Reading observation...")
        data_len = struct.unpack("<I", data_len_bytes)[0]
        data_bytes = self.buffered_socket.read(data_len, True)
        observation_space = observation_space_pb2.ObservationSpaceMessage()
        # print("Parsing observation...")
        observation_space.ParseFromString(data_bytes)
        # print("Parsed observation...")
        return data_len, observation_space

    def send_initial_env(self):
        initial_env = self.initial_env.to_initial_environment_message()
        # print(
        #     "Sending initial environment... ",
        # )
        v = initial_env.SerializeToString()
        # print(base64.b64encode(v).decode())
        self.sock.send(struct.pack("<I", len(v)))
        self.sock.sendall(v)

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        # send the action
        self.last_action = action
        with self.csv_logger.profile("send_action_and_commands"):
            # Translate the action v1 to v2
            if self.action_space_version == ActionSpaceVersion.V1_MINEDOJO:
                translated_action = translate_action_to_v2(action)
            else:
                translated_action = action

            self.ipc.send_action(translated_action, self.queued_commands)
        self.queued_commands.clear()
        # read the response
        self.csv_logger.log("Sent action and reading response...")
        with self.csv_logger.profile("read_response"):
            siz, res = self.read_one_observation()
        self.csv_logger.log("Read observation...")
        final_obs = self.convert_observation_v2(res)

        reward = 0  # Initialize reward to zero
        done = False  # Initialize done flag to False
        truncated = False  # Initialize truncated flag to False
        return (
            final_obs,
            reward,
            done,
            truncated,
            final_obs,
        )

    # when self.render_mode is None: no render is computed
    # when self.render_mode is "human": render returns None, envrionment is already being rendered on the screen
    # when self.render_mode is "rgb_array": render returns the image to be rendered
    # when self.render_mode is "ansi": render returns the text to be rendered
    # when self.render_mode is "rgb_array_list": render returns a list of images to be rendered
    # when self.render_mode is "rgb_tensor": render returns a torch tensor to be rendered
    def render(self) -> Union[RenderFrame, List[RenderFrame], None]:
        return self.observation_converter.render()

    @property
    def render_mode(self) -> Optional[str]:
        return "rgb_array"

    def close(self):
        if not self.use_terminate:
            self.terminate()
        else:
            print("Not terminating the java process")

    def add_command(self, command: str):
        self.queued_commands.append(command)

    def add_commands(self, commands: List[str]):
        self.queued_commands.extend(commands)

    def terminate(self):
        self.ipc.close()
        if self.sock is not None:
            send_exit(self.sock)
            self.sock.close()
            self.sock = None
        print("Terminated the java process")
        pid = self.process.pid if self.process else None
        # wait for the pid to exit
        try:
            if pid:
                os.kill(pid, signal.SIGKILL)
                _, exit_status = os.waitpid(pid, 0)
            else:
                print("No pid to wait for")
        except ChildProcessError:
            print("Child process already terminated")
        print("Terminated the java process")

    @staticmethod
    def get_env_base_path() -> str:
        current_file = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file)
        env_dir = os.path.join(current_dir, "MinecraftEnv", "run")
        return env_dir

    @staticmethod
    def get_env_save_path() -> str:
        return os.path.join(CraftGroundEnvironment.get_env_base_path(), "saves")

    @staticmethod
    def get_env_option_path() -> str:
        return os.path.join(CraftGroundEnvironment.get_env_base_path(), "options.txt")
