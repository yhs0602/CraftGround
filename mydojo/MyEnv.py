import base64
import io
import socket
from typing import Tuple, Optional, Union, List

import gymnasium as gym
import numpy as np
from PIL import Image
from gym.core import ActType, ObsType, RenderFrame

from initial_environment import InitialEnvironment
from json_socket import JSONSocket
from .MyActionSpace import MyActionSpace
from .minecraft import wait_for_server, int_to_action, send_action


class MyEnv(gym.Env):
    def __init__(self, initial_env: InitialEnvironment):
        self.action_space = MyActionSpace(6)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, initial_env.imageSizeX, initial_env.imageSizeY),
            dtype=np.uint8,
        )
        self.state = [0, 0, 0]
        self.initial_env = initial_env

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        sock: socket.socket = wait_for_server()
        # buffered_reader = BufferedReader(sock)
        self.json_socket = JSONSocket(sock)
        self.json_socket.send_json_as_base64(self.initial_env.to_dict())
        print("Sent initial environment")
        print("Reading response...")
        res = self.json_socket.receive_json()  # throw away
        return np.random.rand(
            3, self.initial_env.imageSizeX, self.initial_env.imageSizeY
        ).astype(np.uint8)

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        assert self.action_space.contains(action)  # Check that action is valid
        action_arr = int_to_action(action)

        # send the action
        print("Sending action...")
        send_action(self.json_socket, action_arr)
        # read the response
        print("Reading response...")
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
        print(arr.shape)

        reward = 0  # Initialize reward to zero
        done = False  # Initialize done flag to False
        truncated = False  # Initialize truncated flag to False

        # # Update state based on action
        # if action == 0:
        #     self.state[0] += 0.1
        # elif action == 1:
        #     self.state[1] += 0.1

        # # Check if episode is over
        # if self.state[0] >= 1 or self.state[1] >= 1:
        #     done = True
        #     if self.state[0] >= 1:
        #         reward = 1  # Positive reward if first dimension reaches 1
        #     else:
        #         reward = -1  # Negative reward if second dimension reaches 1

        return (
            arr,
            reward,
            done,
            truncated,
            {},
        )

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        super(MyEnv, self).render()
        print(self.state)
        return None
