import base64
import io
import socket
import struct
import subprocess
from time import sleep
from typing import Tuple, Optional, Union, List

import gymnasium as gym
import numpy as np
from PIL import Image
from gym.core import ActType, ObsType, RenderFrame

from mydojo.initial_environment import InitialEnvironment
from .MyActionSpace import MyActionSpace, MultiActionSpace
from .buffered_socket import BufferedSocket
from .minecraft import wait_for_server, send_action2, send_fastreset2, send_action
from .proto import observation_space_pb2, initial_environment_pb2


class MyEnv(gym.Env):
    def __init__(self, initial_env: InitialEnvironment):
        self.action_space = MyActionSpace(6)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, initial_env.imageSizeX, initial_env.imageSizeY),
            dtype=np.uint8,
        )
        self.initial_env = initial_env
        self.sock = None
        self.buffered_socket = None

    def reset(
        self,
        fast_reset: bool = True,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ):
        if not self.sock:  # first time
            self.start_server()
        else:
            if not fast_reset:
                self.sock.close()
                # wait for server death and restart server
                sleep(5)
                self.start_server()
            else:
                send_fastreset2(self.sock)
                # print("Sent fast reset")
        print("Reading response...")
        res = self.read_one_observation()  # throw away
        return np.random.rand(
            3, self.initial_env.imageSizeX, self.initial_env.imageSizeY
        ).astype(np.uint8)

    def start_server(self):
        subprocess.Popen(
            "./gradlew runClient",
            cwd="/Users/yanghyeonseo/gitprojects/minecraft_env",
            shell=True,
            # stdout=subprocess.DEVNULL,
        )
        sock: socket.socket = wait_for_server()
        self.sock = sock
        self.send_initial_env()
        self.buffered_socket = BufferedSocket(self.sock)
        # self.json_socket.send_json_as_base64(self.initial_env.to_dict())
        print("Sent initial environment")

    def read_one_observation(self) -> ObsType:
        print("Reading observation size...")
        data_len_bytes = self.buffered_socket.read(4, True)
        print("Reading observation...")
        data_len = struct.unpack("<I", data_len_bytes)[0]
        data_bytes = self.buffered_socket.read(data_len, True)
        observation_space = observation_space_pb2.ObservationSpaceMessage()
        print("Parsing observation...")
        observation_space.ParseFromString(data_bytes)
        print("Parsed observation...")
        return observation_space

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
        print(
            "Sending initial environment... ",
        )
        v = initial_env.SerializeToString()
        print(base64.b64encode(v).decode())
        self.sock.send(struct.pack("<I", len(v)))
        self.sock.sendall(v)

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        # send the action
        send_action2(self.sock, action)
        # read the response
        # print("Sent action and reading response...")
        res = self.read_one_observation()
        # save this png byte array to a file
        png_img = base64.b64decode(res.image)  # png byte array
        # decode png byte array to numpy array
        # Create a BytesIO object from the byte array
        bytes_io = io.BytesIO(png_img)

        # Use PIL to open the image from the BytesIO object
        img = Image.open(bytes_io).convert("RGB")

        # Convert the PIL image to a numpy array
        arr = np.array(img)
        arr = np.transpose(arr, (2, 1, 0))

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

        reward = 0  # Initialize reward to one
        done = False  # Initialize done flag to False
        truncated = False  # Initialize truncated flag to False

        return (
            {"obs": res, "rgb": arr},
            reward,
            done,
            truncated,
            {},
        )

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        super(MyEnv, self).render()
        return None


# Deprecated
class MultiDiscreteEnv(MyEnv):
    def __init__(self, initial_env: InitialEnvironment):
        super(MultiDiscreteEnv, self).__init__(initial_env)
        self.action_space = MultiActionSpace([3, 3, 4, 25, 25, 8, 244, 36])
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, initial_env.imageSizeX, initial_env.imageSizeY),
            dtype=np.uint8,
        )
        self.initial_env = initial_env

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        assert self.action_space.contains(action)  # Check that action is valid

        # send the action
        send_action(self.json_socket, action)
        # read the response
        # print("Sent action and reading response...")
        res = self.json_socket.receive_json()
        # save this png byte array to a file
        png_img = base64.b64decode(res["image"])  # png byte array
        # decode png byte array to numpy array
        # Create a BytesIO object from the byte array
        bytes_io = io.BytesIO(png_img)

        # Use PIL to open the image from the BytesIO object
        img = Image.open(bytes_io).convert("RGB")

        # Convert the PIL image to a numpy array
        arr = np.array(img)
        arr = np.transpose(arr, (2, 1, 0))

        # Optionally, you can convert the array to a specific data type, such as uint8
        arr = arr.astype(np.uint8)

        health = res["health"]
        foodLevel = res["foodLevel"]
        saturationLevel = res["saturationLevel"]
        isDead = res["isDead"]
        inventory = res["inventory"]
        soundSubtitles = res["soundSubtitles"]
        # for subtitle in soundSubtitles:
        #     print(f"{subtitle['translateKey']=} {subtitle['x']=} {subtitle['y']=} {subtitle['z']=}")

        reward = 1  # Initialize reward to one
        done = False  # Initialize done flag to False
        truncated = False  # Initialize truncated flag to False

        if isDead:  #
            if self.initial_env.isHardCore:
                reward = -10000000
                done = True
            else:  # send respawn packet
                # pass
                reward = -200
                done = True
                # send_respawn(self.json_socket)
                # res = self.json_socket.receive_json()  # throw away

        # if action == 0:
        #     reward = 1  # Reward of 1 for moving forward

        return (
            arr,
            reward,
            done,
            truncated,
            {},
        )
