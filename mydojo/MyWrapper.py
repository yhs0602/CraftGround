import gymnasium as gym


class MyWrapper(gym.Wrapper):
    def __init__(self, sim):
        super().__init__(env=sim)
