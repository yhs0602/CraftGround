from typing import SupportsFloat, Any, Optional, Union, List

from gymnasium.core import WrapperActType, WrapperObsType

from wrappers.CleanUpFastResetWrapper import CleanUpFastResetWrapper


# Sound wrapper
class RewardTokenChangeWrapper(CleanUpFastResetWrapper):
    def __init__(
        self,
        env,
        token_dim: int,
        reward: Union[float, List[float]],
        reward_once: Union[bool, List[bool]],
        **kwargs,
    ):
        self.env = env
        self.token_dim = token_dim
        self.token_rewarded = [False] * token_dim
        if isinstance(reward, float):
            self.rewards = [reward] * token_dim
        elif isinstance(reward, list):
            self.rewards = reward
        else:
            raise ValueError(f"Invalid reward type: {type(reward)}")
        if isinstance(reward_once, bool):
            self.reward_once = [reward_once] * token_dim
        elif isinstance(reward_once, list):
            self.reward_once = reward_once
        else:
            raise ValueError(f"Invalid reward_once type: {type(reward_once)}")
        self.last_tokens = None
        super().__init__(self.env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        # info_obs = info["obs"]
        token = obs["token"]

        for i in range(self.token_dim):
            can_reward = not self.reward_once[i] or not self.token_rewarded[i]
            # print(
            #     f"{token[i]=} {can_reward=} {self.reward_once[i]=} {self.token_rewarded[i]=}"
            # )
            if (
                token[i] > self.last_tokens[i] and can_reward
            ):  # and not self.token_rewarded[i]:
                reward += self.rewards[i]
                self.token_rewarded[i] = True
                # print(f"Token {i} rewarded")
        self.last_tokens = token
        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )  # , done: deprecated

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        self.token_rewarded = [False] * self.token_dim
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_tokens = obs["token"]
        return obs, info
