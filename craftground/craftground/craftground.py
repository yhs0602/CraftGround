import io
import os
import signal
import socket
import struct
import subprocess
from datetime import datetime
from time import sleep
from typing import Tuple, Optional, Union, List, Any, Dict

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, RenderFrame

from .action_space import ActionSpace
from .buffered_socket import BufferedSocket
from .font import get_font
from .initial_environment import InitialEnvironment
from .minecraft import (
    wait_for_server,
    send_fastreset2,
    send_action_and_commands,
    send_exit,
)
from .proto import observation_space_pb2, initial_environment_pb2


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
        self.sock = None
        self.buffered_socket = None
        self.last_rgb_frames = [None, None]
        self.last_images = [None, None]
        self.last_action = None
        self.render_action = render_action
        self.verbose = verbose
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
            self.start_server(port=self.port)
        else:
            if not fast_reset:
                self.sock.close()
                # wait for server death and restart server
                sleep(5)
                self.start_server(port=self.port)
            else:
                send_fastreset2(self.sock, extra_commands)
                print_with_time("Sent fast reset")
        print_with_time("Reading response...")
        siz, res = self.read_one_observation()
        print_with_time(f"Got response with size {siz}")
        rgb_1, img_1, frame_1 = self.convert_observation(res.image)
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
    ) -> Tuple[np.ndarray, "Image", np.ndarray]:
        # decode png byte array to numpy array
        # Create a BytesIO object from the byte array
        bytes_io = io.BytesIO(png_img)
        # Use PIL to open the image from the BytesIO object
        img = Image.open(bytes_io).convert("RGB")
        # Convert the PIL image to a numpy array
        last_rgb_frame = np.array(img)
        arr = np.transpose(last_rgb_frame, (2, 1, 0))
        # Optionally, you can convert the array to a specific data type, such as uint8
        arr = arr.astype(np.uint8)
        # health = res["health"]
        # foodLevel = res["foodLevel"]
        # saturationLevel = res["saturationLevel"]
        # isDead = res["isDead"]
        # inventory = res["inventory"]
        # soundSubtitles = res["soundSubtitles"]
        # for subtitle in soundSubtitles:
        #     print(f"{subtitle['translateKey']=} {subtitle['x']=} {subtitle['y']=} {subtitle['z']=}")

        return arr, img, last_rgb_frame

    def start_server(self, port=8000):
        my_env = os.environ.copy()
        my_env["PORT"] = str(port)
        self.process = subprocess.Popen(
            "./gradlew runClient",
            cwd=self.env_path,
            shell=True,
            stdout=subprocess.DEVNULL if not self.verbose else None,
            env=my_env,
        )
        sock: socket.socket = wait_for_server(port)
        self.sock = sock
        self.send_initial_env()
        self.buffered_socket = BufferedSocket(self.sock)
        # self.json_socket.send_json_as_base64(self.initial_env.to_dict())
        print_with_time(f"Sent initial environment")

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
        send_action_and_commands(self.sock, action, commands=self.queued_commands)
        self.queued_commands.clear()
        # read the response
        print_with_time("Sent action and reading response...")
        siz, res = self.read_one_observation()
        rgb_1, img_1, frame_1 = self.convert_observation(res.image)
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
        if last_image is None:
            return None

        if self.render_action and self.last_action:
            draw = ImageDraw.Draw(last_image)
            text = self.action_to_symbol(self.last_action)
            position = (0, 0)
            font = get_font()
            font_size = 8
            color = (255, 0, 0)
            draw.text(position, text, font=font, font_size=font_size, fill=color)
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


def print_with_time(*args, **kwargs):
    return  # disable
    time_str = datetime.now().strftime("%H:%M:%S.%f")
    print(f"[{time_str}]", *args, **kwargs)
