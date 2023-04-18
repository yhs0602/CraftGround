from typing import Tuple, Optional, Union, List

import gymnasium as gym
import numpy as np

from gym.core import ActType, ObsType, RenderFrame

from .MyActionSpace import MyActionSpace


class MyEnv(gym.Env):
    def __init__(self):
        self.action_space = MyActionSpace(6)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, 800, 400), dtype="uint8"
        )
        self.state = [0, 0, 0]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.state = [0, 0, 0]
        return np.random.rand(3, 800, 400).astype(np.float32)

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        assert self.action_space.contains(action)  # Check that action is valid

        reward = 0  # Initialize reward to zero
        done = False  # Initialize done flag to False
        truncated = False  # Initialize truncated flag to False

        # Update state based on action
        if action == 0:
            self.state[0] += 0.1
        elif action == 1:
            self.state[1] += 0.1

        # Check if episode is over
        if self.state[0] >= 1 or self.state[1] >= 1:
            done = True
            if self.state[0] >= 1:
                reward = 1  # Positive reward if first dimension reaches 1
            else:
                reward = -1  # Negative reward if second dimension reaches 1

        return (
            np.random.rand(3, 800, 400).astype(np.float32),
            reward,
            done,
            truncated,
            {},
        )

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        super(MyEnv, self).render()
        print(self.state)
        return None
