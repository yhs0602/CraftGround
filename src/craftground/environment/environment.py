import os
import re
import shutil
import signal
import subprocess
import sys
import threading
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union
import weakref

import gymnasium as gym
import numpy as np
import torch
from gymnasium.core import ActType, RenderFrame

from ..constants import PROCESS_TERMINATION_TIMEOUT
from ..csv_logger import CsvLogger, LogBackend
from ..exceptions import ProcessTerminationError
from ..initial_environment_config import InitialEnvironmentConfig
from ..proto.observation_space_pb2 import ObservationSpaceMessage
from .action_space import (
    ActionSpaceVersion,
    action_v2_dict_to_message,
    declare_action_space,
    translate_action_to_v2,
)
from .observation_converter import ObservationConverter
from .observation_space import declare_observation_space
from .socket_ipc import SocketIPC


class ObservationTensorType(Enum):
    NONE = 0
    CUDA_DLPACK = 1
    APPLE_TENSOR = 2


class TypedObservation(TypedDict):
    full: ObservationSpaceMessage
    pov: Union[np.ndarray, torch.Tensor]
    rgb: Union[np.ndarray, torch.Tensor]
    pov_2: Union[np.ndarray, torch.Tensor]
    rgb_2: Union[np.ndarray, torch.Tensor]


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
        no_threaded_optimizations: bool = True,
        track_native_memory: bool = False,
        ld_preload: Optional[str] = None,
        native_debug: bool = False,
        verbose_python: bool = False,
        verbose_gradle: bool = False,
        verbose_jvm: bool = False,
        profile: bool = False,
        profile_jni: bool = False,
    ):
        self.action_space_version = action_space_version
        self.action_space = declare_action_space(action_space_version)
        self.observation_space = declare_observation_space(
            initial_env.imageSizeX, initial_env.imageSizeY
        )
        self.initial_env = initial_env
        self.initial_env_message = initial_env.to_initial_environment_message()

        self.use_terminate = use_terminate
        self.cleanup_world = cleanup_world
        self.use_vglrun = use_vglrun
        self.no_threaded_optimizations = no_threaded_optimizations
        self.native_debug = native_debug
        self.track_native_memory = track_native_memory
        self.ld_preload = ld_preload
        self.encoding_mode = initial_env.screen_encoding_mode

        self.last_action = None
        self.render_action = render_action

        self.verbose = verbose
        self.verbose_python = verbose_python
        self.verbose_gradle = verbose_gradle
        self.verbose_jvm = verbose_jvm
        self.profile = profile
        self.profile_jni = profile_jni

        self.render_alternating_eyes = render_alternating_eyes
        self.render_alternating_eyes_counter = 0
        self.find_free_port = find_free_port
        self.queued_commands = []

        self.process = None
        self.use_shared_memory = use_shared_memory
        if env_path is None:
            self.env_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "MinecraftEnv",
            )
        else:
            self.env_path = env_path
            gradle_path = shutil.which("gradlew", path=self.env_path)
            if gradle_path is None:
                raise FileNotFoundError(
                    f"eXecutable gradlew not found in {self.env_path}. Please provide the correct path to the environment."
                )

        self.logger = CsvLogger(
            "py_log.csv",
            profile=profile,
            backend=LogBackend.BOTH if verbose_python else LogBackend.NONE,
        )

        if self.use_shared_memory:
            from .boost_ipc import BoostIPC  # type: ignore

            self.ipc = BoostIPC(
                port,
                find_free_port,
                self.initial_env_message,
                self.logger,
            )
        else:
            self.ipc = SocketIPC(
                self.logger,
                self.initial_env_message,
                port,
                find_free_port,
            )

        self.observation_converter = ObservationConverter(
            self.initial_env.screen_encoding_mode,
            self.initial_env.imageSizeX,
            self.initial_env.imageSizeY,
            self.logger,
            self.initial_env.eye_distance > 0,
            self.render_action,
            self.render_mode,
        )

        weakref.finalize(self, self.close)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[TypedObservation, TypedObservation]:
        """Reset the environment to an initial state.

        Args:
            seed: Optional random seed for environment initialization
            options: Optional dictionary containing:
                - fast_reset: If True, use fast reset (default: True)
                - extra_commands: List of additional commands to execute on reset

        Returns:
            Tuple of (observation, info) where both are the same TypedObservation
        """
        if options is None:
            options = {}
        fast_reset = options.get("fast_reset", True)
        extra_commands = options.get("extra_commands", [])

        self.ensure_alive(fast_reset, extra_commands, seed)

        with self.logger.profile("read_response"):
            res = self.ipc.read_observation()
        final_obs = self.convert_observation_v2(res)
        return final_obs, final_obs

    def convert_observation_v2(
        self, res: ObservationSpaceMessage
    ) -> Dict[str, Union[np.ndarray, "torch.Tensor", ObservationSpaceMessage]]:
        """Convert observation space message to dictionary format.

        Args:
            res: Protocol buffer observation space message

        Returns:
            Dictionary containing:
            - 'full': Original ObservationSpaceMessage
            - 'pov': First viewpoint RGB image (numpy array or torch.Tensor)
            - 'rgb': Same as 'pov' (for backward compatibility)
            - 'pov_2': Second viewpoint RGB image (if binocular vision enabled)
            - 'rgb_2': Same as 'pov_2' (for backward compatibility)
        """
        with self.logger.profile("convert_observation"):
            rgb_1, rgb_2 = self.observation_converter.convert(res)

        self.queued_commands = []
        res.yaw = ((res.yaw + 180) % 360) - 180
        final_obs: Dict[
            str, Union[np.ndarray, "torch.Tensor", ObservationSpaceMessage]
        ] = {
            "full": res,
            "pov": rgb_1,
            "rgb": rgb_1,
        }
        if rgb_2 is not None:
            final_obs["pov_2"] = rgb_2
            final_obs["rgb_2"] = rgb_2
        return final_obs

    @property
    def is_alive(self) -> bool:
        if self.process is None:
            return False
        exit_code = self.process.poll()
        if exit_code is not None:
            self.logger.log(f"Java process exited with code {exit_code}")
            return False
        return True

    # (alive and fast_reset) -> send fast reset
    # (alive and not fast_reset) -> destroy and start server
    # (not alive) -> start server
    def ensure_alive(self, fast_reset, extra_commands, seed: int):
        if self.is_alive:
            if fast_reset:
                self.ipc.send_fastreset2(extra_commands)
                return
            else:
                self.terminate()

        if self.use_shared_memory:
            from .boost_ipc import BoostIPC  # type: ignore

            self.ipc = BoostIPC(
                self.ipc.port,
                self.ipc.find_free_port,
                self.initial_env_message,
                self.logger,
            )
        else:
            self.ipc = SocketIPC(
                self.logger,
                self.initial_env_message,
                self.ipc.port,
                self.ipc.find_free_port,
            )

        self.start_server(seed=seed)

    def start_server(self, seed: int):
        # Remove orphan java processes
        self.ipc.remove_orphan_java_processes()
        # Prepare command
        my_env = os.environ.copy()
        my_env["PORT"] = str(self.ipc.port)
        my_env["USE_SHARED_MEMORY"] = str(int(self.use_shared_memory))
        my_env["VERBOSE"] = str(int(self.verbose_jvm))
        if self.track_native_memory:
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

        if os.name == "nt":
            cmd = f".\\gradlew runClient -w --no-daemon"
        else:
            cmd = f"./gradlew runClient -w --no-daemon"  #  --args="--width {self.initial_env.imageSizeX} --height {self.initial_env.imageSizeY}"'
        if self.use_vglrun:
            cmd = f"vglrun {cmd}"
            if self.no_threaded_optimizations:  # __GL_THREADED_OPTIMIZATIONS=0
                cmd = f"__GL_THREADED_OPTIMIZATIONS=0 {cmd}"
        if self.ld_preload:
            my_env["LD_PRELOAD"] = self.ld_preload
        if self.profile_jni:
            my_env["CRAFTGROUND_NATIVE_PROFILE"] = "1"
        self.logger.log(f"Starting server with command: {cmd}")

        # Launch the server
        kwargs = dict(
            cwd=self.env_path,
            shell=True,
            stdout=subprocess.DEVNULL if not self.verbose_gradle else None,
            env=my_env,
        )

        if hasattr(os, "setsid"):
            # Linux / Unix
            kwargs["preexec_fn"] = os.setsid
        elif sys.platform == "win32":
            # Windows
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        self.process = subprocess.Popen(cmd, **kwargs)
        self.server_event = threading.Event()
        threading.Thread(
            target=self.monitor_process,
            args=(self.process, self.server_event),
            daemon=True,
        ).start()

        self.ipc.start_communication(self.server_event)

    def monitor_process(self, process, server_event):
        process.wait()
        self.logger.log(f"Java process exited with code {process.returncode}")
        server_event.set()

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

    def step(self, action: ActType) -> Tuple[
        TypedObservation,
        float,
        bool,
        bool,
        TypedObservation,
    ]:
        """Run one timestep of the environment's dynamics.

        Args:
            action: Action to take in the environment

        Returns:
            Tuple containing:
            - observation: Current observation
            - reward: Reward from the action (always 0, to be implemented by user)
            - done: Whether the episode has ended (always False)
            - truncated: Whether the episode was truncated (always False)
            - info: Additional information (same as observation)
        """
        # send the action
        self.last_action = action
        with self.logger.profile("send_action_and_commands"):
            # Translate the action v1 to v2
            if self.action_space_version == ActionSpaceVersion.V1_MINEDOJO:
                translated_action = translate_action_to_v2(action)
            else:
                translated_action = action

            self.ipc.send_action(
                action_v2_dict_to_message(translated_action), self.queued_commands
            )
        self.queued_commands.clear()
        # read the response
        self.logger.log("Sent action and reading response...")
        with self.logger.profile("read_response"):
            res = self.ipc.read_observation()
        self.logger.log("Read observation...")
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
        """Terminate the Java process safely.

        Raises:
            ProcessTerminationError: If process termination fails
        """
        self.server_event = None

        # Clean up IPC connection
        try:
            self.ipc.destroy()
        except Exception as e:
            self.logger.log(f"IPC destroy failed: {e}")

        p = self.process
        if not p:
            self.logger.log("No process to terminate")
            return

        pid = p.pid

        try:
            if hasattr(os, "getpgid"):
                # Unix: Process group kill
                self._terminate_unix(p, pid)
            else:
                # Windows
                self._terminate_windows(p, pid)
        except ProcessTerminationError:
            # Force kill as last resort
            self._force_kill(p, pid)
        except Exception as e:
            self.logger.log(f"Terminate error: {e}")
            self._force_kill(p, pid)
        finally:
            self.process = None
            self.logger.log("Terminated the java process")

    def _terminate_unix(self, p: subprocess.Popen, pid: int):
        """Terminate process on Unix systems.

        Args:
            p: Process object
            pid: Process ID

        Raises:
            ProcessTerminationError: If termination fails
        """
        try:
            pgrp = os.getpgid(pid)
            os.killpg(pgrp, signal.SIGTERM)
            self.logger.log(f"Sent SIGTERM to process group {pgrp}(pid {pid})")
        except (OSError, ProcessLookupError) as e:
            raise ProcessTerminationError(f"SIGTERM failed: {e}") from e

        try:
            p.wait(timeout=PROCESS_TERMINATION_TIMEOUT)
        except subprocess.TimeoutExpired:
            self.logger.log("Wait timeout; sending SIGKILL")
            try:
                pgrp = os.getpgid(pid)
                os.killpg(pgrp, signal.SIGKILL)
            except (OSError, ProcessLookupError) as e:
                raise ProcessTerminationError(f"SIGKILL failed: {e}") from e

    def _terminate_windows(self, p: subprocess.Popen, pid: int):
        """Terminate process on Windows systems.

        Args:
            p: Process object
            pid: Process ID

        Raises:
            ProcessTerminationError: If termination fails
        """
        try:
            # Popen with CREATE_NEW_PROCESS_GROUP uses CTRL_BREAK_EVENT
            p.send_signal(signal.CTRL_BREAK_EVENT)
            self.logger.log(f"Sent CTRL_BREAK_EVENT to pid {pid}")
        except (OSError, ProcessLookupError) as e:
            self.logger.log(f"CTRL_BREAK_EVENT failed: {e}; calling terminate()")
            try:
                p.terminate()
            except Exception as e2:
                raise ProcessTerminationError(f"terminate() failed: {e2}") from e2

        try:
            p.wait(timeout=PROCESS_TERMINATION_TIMEOUT)
        except subprocess.TimeoutExpired:
            self.logger.log("Wait timeout; calling kill()")
            try:
                p.kill()
            except Exception as e:
                raise ProcessTerminationError(f"kill() failed: {e}") from e

    def _force_kill(self, p: subprocess.Popen, pid: int):
        """Force kill process as last resort.

        Args:
            p: Process object
            pid: Process ID
        """
        try:
            if sys.platform == "win32":
                p.kill()
            else:
                os.kill(pid, signal.SIGKILL)
            self.logger.log(f"Force killed process {pid}")
        except (OSError, ProcessLookupError) as e:
            self.logger.log(f"Force kill failed: {e}")

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

    def __del__(self):
        self.close()
        if self.logger:
            self.logger.close()
