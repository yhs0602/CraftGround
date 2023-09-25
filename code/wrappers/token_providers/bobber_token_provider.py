from wrappers.token_providers.base_token_provider import BaseTokenProvider


class BobberTokenProvider(BaseTokenProvider):
    def __init__(self, token_idx: int, **kwargs):
        super().__init__(token_dim=1, token_idx=token_idx, **kwargs)

    def provide_token_step(self, obs, info, token):
        obs_info = info["obs"]
        token[self.token_idx] = obs_info.bobber_thrown
        super().provide_token_step(obs, info, token)
