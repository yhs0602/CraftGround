import io
import os
import signal
import socket
import struct
import subprocess
from time import sleep
from typing import Tuple, Optional, Union, List, Any, Dict

import gymnasium as gym
import numpy as np
import psutil
from PIL import Image, ImageDraw
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, RenderFrame

from .action_space import ActionSpace
from .buffered_socket import BufferedSocket
from .csv_logger import CsvLogger
from .font import get_font
from .initial_environment import InitialEnvironment
from .minecraft import (
    wait_for_server,
    send_fastreset2,
    send_action_and_commands,
    send_exit,
)
from .print_with_time import print_with_time
from .proto import observation_space_pb2, initial_environment_pb2
from .screen_encoding_modes import ScreenEncodingMode


class CraftGroundEnvironment(gym.Env):
    def __init__(
        self,
        initial_env: InitialEnvironment,
        verbose=False,
        env_path=None,
        port=8000,
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
        self.action_space = ActionSpace(6)
        entity_info_space = gym.spaces.Dict(
            {
                "unique_name": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.int32,
                ),
                "translation_key": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.int32,
                ),
                "x": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float64,
                ),
                "y": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float64,
                ),
                "z": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float64,
                ),
                "yaw": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float64,
                ),
                "pitch": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float64,
                ),
                "health": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float64,
                ),
            }
        )
        sound_entry_space = gym.spaces.Dict(
            {
                "translate_key": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.int32
                ),
                "x": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
                "y": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
                "z": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
                "age": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.int32),
            }
        )
        entities_within_distance_space = gym.spaces.Sequence(entity_info_space)
        status_effect_space = gym.spaces.Dict(
            {
                "translation_key": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.int32
                ),
                "amplifier": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.int32
                ),
                "duration": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.int32
                ),
            }
        )
        self.observation_space = gym.spaces.Dict(
            {
                "obs": spaces.Dict(
                    {
                        "image": spaces.Box(
                            low=0,
                            high=255,
                            shape=(initial_env.imageSizeX, initial_env.imageSizeY, 3),
                            dtype=np.uint8,
                        ),
                        "position": spaces.Box(
                            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                        ),
                        "yaw": spaces.Box(
                            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64
                        ),
                        "pitch": spaces.Box(
                            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64
                        ),
                        "health": spaces.Box(
                            low=0, high=np.inf, shape=(1,), dtype=np.float64
                        ),
                        "food_level": spaces.Box(
                            low=0, high=np.inf, shape=(1,), dtype=np.float64
                        ),
                        "saturation_level": spaces.Box(
                            low=0, high=np.inf, shape=(1,), dtype=np.float64
                        ),
                        "is_dead": spaces.Discrete(2),
                        "inventory": spaces.Sequence(
                            spaces.Dict(
                                {
                                    "raw_id": spaces.Box(
                                        low=-np.inf,
                                        high=np.inf,
                                        shape=(1,),
                                        dtype=np.int32,
                                    ),
                                    "translation_key": spaces.Box(
                                        low=-np.inf,
                                        high=np.inf,
                                        shape=(1,),
                                        dtype=np.int32,
                                    ),
                                    "count": spaces.Box(
                                        low=-np.inf,
                                        high=np.inf,
                                        shape=(1,),
                                        dtype=np.int32,
                                    ),
                                    "durability": spaces.Box(
                                        low=-np.inf,
                                        high=np.inf,
                                        shape=(1,),
                                        dtype=np.int32,
                                    ),
                                    "max_durability": spaces.Box(
                                        low=-np.inf,
                                        high=np.inf,
                                        shape=(1,),
                                        dtype=np.int32,
                                    ),
                                }
                            ),
                        ),
                        "raycast_result": spaces.Dict(
                            {
                                "type": spaces.Discrete(3),
                                "target_block": spaces.Dict(
                                    {
                                        "x": spaces.Box(
                                            low=-np.inf,
                                            high=np.inf,
                                            shape=(1,),
                                            dtype=np.int32,
                                        ),
                                        "y": spaces.Box(
                                            low=-np.inf,
                                            high=np.inf,
                                            shape=(1,),
                                            dtype=np.int32,
                                        ),
                                        "z": spaces.Box(
                                            low=-np.inf,
                                            high=np.inf,
                                            shape=(1,),
                                            dtype=np.int32,
                                        ),
                                        "translation_key": spaces.Box(
                                            low=-np.inf,
                                            high=np.inf,
                                            shape=(1,),
                                            dtype=np.int32,
                                        ),
                                    }
                                ),
                                "target_entity": entity_info_space,
                            }
                        ),
                        "sound_subtitles": spaces.Sequence(sound_entry_space),
                        "status_effects": spaces.Sequence(status_effect_space),
                        "killed_statistics": spaces.Dict(),
                        "mined_statistics": spaces.Dict(),
                        "misc_statistics": spaces.Dict(),
                        "visible_entities": spaces.Sequence(entity_info_space),
                        "surrounding_entities": spaces.Dict(),  # you need to decide how to handle this
                        "bobber_thrown": spaces.Discrete(2),
                        "experience": spaces.Box(
                            low=0, high=np.inf, shape=(1,), dtype=np.int32
                        ),
                        "world_time": spaces.Box(
                            low=-np.inf, high=np.inf, shape=(1,), dtype=np.int64
                        ),
                        "last_death_message": spaces.Text(
                            min_length=0, max_length=1000
                        ),
                        "image_2": spaces.Box(
                            low=0,
                            high=255,
                            shape=(initial_env.imageSizeX, initial_env.imageSizeY, 3),
                            dtype=np.uint8,
                        ),
                    }
                ),
            }
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
        self.last_rgb_frames = [None, None]
        self.last_images = [None, None]
        self.last_action = None
        self.render_action = render_action
        self.verbose = verbose
        self.verbose_python = verbose_python
        self.verbose_gradle = verbose_gradle
        self.verbose_jvm = verbose_jvm
        self.profile = profile

        self.render_alternating_eyes = render_alternating_eyes
        self.render_alternating_eyes_counter = 0
        self.port = port
        self.queued_commands = []
        self.process = None
        if env_path is None:
            self.env_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "MinecraftEnv",
            )
        else:
            self.env_path = env_path
        self.csv_logger = CsvLogger(
            "py_log.csv", enabled=verbose and False, profile=profile
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        if options is None:
            options = {}
        fast_reset = options.get("fast_reset", False)
        extra_commands = options.get("extra_commands", [])
        if not self.sock:  # first time
            self.start_server(
                port=self.port,
                use_vglrun=self.use_vglrun,
                track_native_memory=self.track_native_memory,
                ld_preload=self.ld_preload,
            )
        else:
            if not fast_reset:
                self.sock.close()
                self.terminate()
                # wait for server death and restart server
                sleep(5)
                self.start_server(
                    port=self.port,
                    use_vglrun=self.use_vglrun,
                    track_native_memory=self.track_native_memory,
                    ld_preload=self.ld_preload,
                )
            else:
                self.csv_logger.profile_start("fast_reset")
                send_fastreset2(self.sock, extra_commands)
                self.csv_logger.profile_end("fast_reset")
                self.csv_logger.log("Sent fast reset")
                if self.verbose_python:
                    print_with_time("Sent fast reset")
        if self.verbose_python:
            print_with_time("Reading response...")
        self.csv_logger.log("Reading response...")
        self.csv_logger.profile_start("read_response")
        siz, res = self.read_one_observation()
        self.csv_logger.profile_end("read_response")
        if self.verbose_python:
            print_with_time(f"Got response with size {siz}")
        self.csv_logger.log(f"Got response with size {siz}")
        self.csv_logger.profile_start("convert_observation")
        rgb_1, img_1, frame_1 = self.convert_observation(res.image)
        self.csv_logger.profile_end("convert_observation")
        rgb_2 = None
        img_2 = None
        frame_2 = None
        if res.image_2 is not None and res.image_2 != b"":
            rgb_2, img_2, frame_2 = self.convert_observation(res.image_2)
        self.queued_commands = []
        final_obs = {
            "obs": res,
            "rgb": rgb_1,
        }
        self.last_images = [img_1, img_2]
        self.last_rgb_frames = [frame_1, frame_2]
        if rgb_2 is not None:
            final_obs["rgb_2"] = rgb_2
        return final_obs, final_obs

    def convert_observation(
        self, png_img: bytes
    ) -> Tuple[np.ndarray, Optional["Image"], np.ndarray]:
        if self.encoding_mode == ScreenEncodingMode.PNG:
            # decode png byte array to numpy array
            # Create a BytesIO object from the byte array
            self.csv_logger.profile_start("convert_observation/decode_png")
            bytes_io = io.BytesIO(png_img)
            # Use PIL to open the image from the BytesIO object
            img = Image.open(bytes_io).convert("RGB")
            # Flip y axis
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            self.csv_logger.profile_end("convert_observation/decode_png")
            self.csv_logger.profile_start("convert_observation/convert_to_numpy")
            # Convert the PIL image to a numpy array
            last_rgb_frame = np.array(img)
            arr = np.transpose(last_rgb_frame, (2, 1, 0))
            # Optionally, you can convert the array to a specific data type, such as uint8
            arr = arr.astype(np.uint8)
            self.csv_logger.profile_end("convert_observation/convert_to_numpy")
        elif self.encoding_mode == ScreenEncodingMode.RAW:
            # decode raw byte array to numpy array
            self.csv_logger.profile_start("convert_observation/decode_raw")
            last_rgb_frame = np.frombuffer(png_img, dtype=np.uint8).reshape(
                (self.initial_env.imageSizeY, self.initial_env.imageSizeX, 3)
            )
            # Flip y axis using np
            last_rgb_frame = np.flip(last_rgb_frame, axis=0)
            arr = np.transpose(last_rgb_frame, (2, 1, 0))  # channels, width, height
            img = None
            self.csv_logger.profile_end("convert_observation/decode_raw")
        else:
            raise ValueError(f"Unknown encoding mode: {self.encoding_mode}")

        # health = res["health"]
        # foodLevel = res["foodLevel"]
        # saturationLevel = res["saturationLevel"]
        # isDead = res["isDead"]
        # inventory = res["inventory"]
        # soundSubtitles = res["soundSubtitles"]
        # for subtitle in soundSubtitles:
        #     print(f"{subtitle['translateKey']=} {subtitle['x']=} {subtitle['y']=} {subtitle['z']=}")

        return arr, img, last_rgb_frame

    def start_server(
        self,
        port: int,
        use_vglrun: bool,
        track_native_memory: bool,
        ld_preload: Optional[str],
    ):
        self.remove_orphan_java_processes()
        # Check if a file exists
        socket_path = f"/tmp/minecraftrl_{port}.sock"
        if os.path.exists(socket_path):
            raise FileExistsError(
                f"Socket file {socket_path} already exists. Please choose another port."
            )
        my_env = os.environ.copy()
        my_env["PORT"] = str(port)
        my_env["VERBOSE"] = str(int(self.verbose_jvm))
        if track_native_memory:
            my_env["CRAFTGROUND_JVM_NATIVE_TRACKING"] = "detail"
        if self.native_debug:
            my_env["CRAFGROUND_NATIVE_DEBUG"] = "True"
        cmd = "./gradlew runClient -w --no-daemon"
        if use_vglrun:
            cmd = f"vglrun {cmd}"
        if ld_preload:
            my_env["LD_PRELOAD"] = ld_preload
        print(f"{cmd=}")
        self.process = subprocess.Popen(
            cmd,
            cwd=self.env_path,
            shell=True,
            stdout=subprocess.DEVNULL if not self.verbose_gradle else None,
            env=my_env,
        )
        sock: socket.socket = wait_for_server(port)
        self.sock = sock
        self.send_initial_env()
        self.buffered_socket = BufferedSocket(self.sock)
        # self.json_socket.send_json_as_base64(self.initial_env.to_dict())
        if self.verbose_python:
            print_with_time(f"Sent initial environment")
        self.csv_logger.log(f"Sent initial environment")

    def read_one_observation(self) -> (int, ObsType):
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
        initial_env = initial_environment_pb2.InitialEnvironmentMessage()
        initial_env.initialInventoryCommands.extend(
            self.initial_env.initialInventoryCommands
        )
        if self.initial_env.initialPosition is not None:
            initial_env.initialPosition.extend(self.initial_env.initialPosition)
        initial_env.initialMobsCommands.extend(self.initial_env.initialMobsCommands)
        initial_env.imageSizeX = self.initial_env.imageSizeX
        initial_env.imageSizeY = self.initial_env.imageSizeY
        initial_env.seed = self.initial_env.seed
        initial_env.allowMobSpawn = self.initial_env.allowMobSpawn
        initial_env.alwaysNight = self.initial_env.alwaysNight
        initial_env.alwaysDay = self.initial_env.alwaysDay
        initial_env.initialWeather = self.initial_env.initialWeather
        initial_env.isWorldFlat = self.initial_env.isWorldFlat
        initial_env.visibleSizeX = self.initial_env.visibleSizeX
        initial_env.visibleSizeY = self.initial_env.visibleSizeY
        if self.initial_env.initialExtraCommands is not None:
            initial_env.initialExtraCommands.extend(
                self.initial_env.initialExtraCommands
            )
        if self.initial_env.killedStatKeys is not None:
            initial_env.killedStatKeys.extend(self.initial_env.killedStatKeys)
        if self.initial_env.minedStatKeys is not None:
            initial_env.minedStatKeys.extend(self.initial_env.minedStatKeys)
        if self.initial_env.miscStatKeys is not None:
            initial_env.miscStatKeys.extend(self.initial_env.miscStatKeys)
        if self.initial_env.surrounding_entities_keys is not None:
            initial_env.surroundingEntityDistances.extend(
                self.initial_env.surrounding_entities_keys
            )
        initial_env.hudHidden = self.initial_env.isHudHidden
        initial_env.render_distance = (
            self.initial_env.render_distance
            if self.initial_env.render_distance is not None
            else 6
        )
        initial_env.simulation_distance = (
            self.initial_env.simulation_distance
            if self.initial_env.simulation_distance is not None
            else 6
        )
        initial_env.biocular = self.initial_env.is_biocular
        initial_env.eye_distance = self.initial_env.eye_distance
        initial_env.structurePaths.extend(self.initial_env.structure_paths)
        initial_env.noWeatherCycle = self.initial_env.noWeatherCycle
        initial_env.no_pov_effect = self.initial_env.no_pov_effect
        initial_env.noTimeCycle = self.initial_env.noTimeCycle
        initial_env.request_raycast = self.initial_env.request_raycast
        initial_env.screen_encoding_mode = self.initial_env.screen_encoding_mode.value
        initial_env.requiresSurroundingBlocks = (
            self.initial_env.requiresSurroundingBlocks
        )
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
        self.csv_logger.profile_start("send_action_and_commands")
        send_action_and_commands(
            self.sock,
            action,
            commands=self.queued_commands,
            verbose=self.verbose_python,
        )
        self.csv_logger.profile_end("send_action_and_commands")
        self.queued_commands.clear()
        # read the response
        if self.verbose_python:
            print_with_time("Sent action and reading response...")
        self.csv_logger.log("Sent action and reading response...")
        self.csv_logger.profile_start("read_response")
        siz, res = self.read_one_observation()
        self.csv_logger.profile_end("read_response")
        if self.verbose_python:
            print_with_time("Read observation...")
        self.csv_logger.log("Read observation...")
        self.csv_logger.profile_start("convert_observation")
        rgb_1, img_1, frame_1 = self.convert_observation(res.image)
        self.csv_logger.profile_end("convert_observation")
        rgb_2 = None
        img_2 = None
        frame_2 = None
        if res.image_2 is not None and res.image_2 != b"":
            rgb_2, img_2, frame_2 = self.convert_observation(res.image_2)
        final_obs = {
            "obs": res,
            "rgb": rgb_1,
        }
        self.last_images = [img_1, img_2]
        self.last_rgb_frames = [frame_1, frame_2]
        if rgb_2 is not None:
            final_obs["rgb_2"] = rgb_2

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

    def render(self) -> Union[RenderFrame, List[RenderFrame], None]:
        # print("Rendering...")
        # select last_image and last_frame
        if self.render_alternating_eyes:
            last_image = self.last_images[self.render_alternating_eyes_counter]
            last_rgb_frame = self.last_rgb_frames[self.render_alternating_eyes_counter]
            self.render_alternating_eyes_counter = (
                1 - self.render_alternating_eyes_counter
            )
        else:
            last_image = self.last_images[0]
            last_rgb_frame = self.last_rgb_frames[0]
        if last_image is None and last_rgb_frame is None:
            return None
        if self.render_action and self.last_action:
            if last_image is None:
                last_image = Image.fromarray(last_rgb_frame)
            self.csv_logger.profile_start("render_action")
            draw = ImageDraw.Draw(last_image)
            text = self.action_to_symbol(self.last_action)
            position = (0, 0)
            font = get_font()
            font_size = 8
            color = (255, 0, 0)
            draw.text(position, text, font=font, font_size=font_size, fill=color)
            self.csv_logger.profile_end("render_action")
            return np.array(last_image)
        else:
            return last_rgb_frame

    def action_to_symbol(self, action) -> str:
        res = ""
        if action[0] == 1:
            res += "↑"
        elif action[0] == 2:
            res += "↓"
        if action[1] == 1:
            res += "←"
        elif action[1] == 2:
            res += "→"
        if action[2] == 1:
            res += "jump"  # "⤴"
        elif action[2] == 2:
            res += "sneak"  # "⤵"
        elif action[2] == 3:
            res += "sprint"  # "⚡"
        if action[3] > 12:  # pitch up
            res += "⤒"
        elif action[3] < 12:  # pitch down
            res += "⤓"
        if action[4] > 12:  # yaw right
            res += "⏭"
        elif action[4] < 12:  # yaw left
            res += "⏮"
        if action[5] == 1:  # use
            res += "use"  # "⚒"
        elif action[5] == 2:  # drop
            res += "drop"  # "🤮"
        elif action[5] == 3:  # attack
            res += "attack"  # "⚔"
        return res

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

    def remove_orphan_java_processes(self):
        print("Removing orphan Java processes...")
        target_directory = "/tmp"
        file_pattern = "minecraftrl_"
        file_usage = {}
        no_such_processes = 0
        access_denied_processes = 0
        # 파일 사용 정보 수집
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                for file in proc.open_files():
                    if (
                        file.path.startswith(target_directory)
                        and file_pattern in file.path
                    ):
                        if file.path not in file_usage:
                            file_usage[file.path] = []
                        file_usage[file.path].append(proc.info)
            except psutil.NoSuchProcess:
                no_such_processes += 1
                continue
            except psutil.AccessDenied:
                access_denied_processes += 1
                continue
            except Exception as e:
                print(f"Error: {e}")
                continue

        # 파일 사용하는 모든 프로세스가 Java인지 확인 및 처리
        for file_path, processes in file_usage.items():
            if all(proc["name"].lower() == "java" for proc in processes):
                for proc in processes:
                    os.kill(proc["pid"], signal.SIGTERM)
                    print(f"Killed Java process {proc['pid']} using file {file_path}")
                os.remove(file_path)
                print(f"Removed {file_path}")
        print(
            f"Removed orphan Java processes: {access_denied_processes} access denied, {no_such_processes} no such process"
        )
