from typing import SupportsFloat, Any

from gymnasium.core import WrapperActType, WrapperObsType

from wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


# Sound wrapper
class RewardTokenChangeWrapper(CleanUpFastResetWrapper):
    def __init__(self, env, token_dim: int, reward: float, **kwargs):
        self.env = env
        self.token_dim = token_dim
        self.token_rewarded = [False] * token_dim
        self.reward = reward
        super().__init__(self.env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info_obs = info["obs"]
        token = info["token"]

        for i in range(self.token_dim):
            if token[i] and not self.token_rewarded[i]:
                reward += self.reward
                self.token_rewarded[i] = True
                print(f"Token {i} rewarded")

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )  # , done: deprecated
