from typing import Any, Optional

import gymnasium as gym


class FastResetWrapper(gym.Wrapper):
    def __init__(self, env, **kwargs):
        self.env = env
        super().__init__(self.env)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        if options is None:
            options = {}
        options["fast_reset"] = True
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info
