import base64
import io
import socket
import subprocess
from time import sleep
from typing import Tuple, Optional, Union, List

import gymnasium as gym
import numpy as np
from PIL import Image
from gym.core import ActType, ObsType, RenderFrame

from mydojo.initial_environment import InitialEnvironment
from mydojo.json_socket import JSONSocket
from .MyActionSpace import MyActionSpace, MultiActionSpace
from .minecraft import wait_for_server, send_action, send_respawn, send_fastreset


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
        self.json_socket = None

    def reset(
        self,
        fast_reset: bool = True,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ):
        if not self.json_socket:  # first time
            self.start_server()
        else:
            if not fast_reset:
                self.json_socket.close()
                # wait for server death and restart server
                sleep(5)
                self.start_server()
            else:
                send_fastreset(self.json_socket)
                print("Sent fast reset")
        print("Reading response...")
        res = self.json_socket.receive_json()  # throw away
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
        self.json_socket = JSONSocket(sock)
        self.json_socket.send_json_as_base64(self.initial_env.to_dict())
        print("Sent initial environment")

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
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

        res["rgb"] = arr
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
            res,
            reward,
            done,
            truncated,
            {},
        )

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        super(MyEnv, self).render()
        return None


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
        self.json_socket = None

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
