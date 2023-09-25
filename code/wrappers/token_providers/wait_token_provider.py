from wrappers.token_providers.base_token_provider import BaseTokenProvider


class WaitTokenProvider(BaseTokenProvider):
    def __init__(self, token_idx: int, total_steps: int, **kwargs):
        super().__init__(token_dim=1, token_idx=token_idx, **kwargs)
        self.total_steps = total_steps
        self.base_time = 0

    def provide_token_step(self, obs, info, token):
        obs_info = info["obs"]
        bobber_thrown = obs_info.bobber_thrown
        if not bobber_thrown:
            self.base_time = obs_info.world_time
        token[self.token_idx] = (
            obs_info.world_time - self.base_time
        ) / self.total_steps
        super().provide_token_step(obs, info, token)

    def provide_token_reset(self, obs, info, token):
        obs_info = info["obs"]
        self.base_time = obs_info.world_time
        token[self.token_idx] = 0
        super().provide_token_reset(obs, info, token)
