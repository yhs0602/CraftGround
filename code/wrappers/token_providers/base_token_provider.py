class BaseTokenProvider:
    def __init__(self, token_dim: int, token_idx: int, **kwargs):
        self.token_dim = token_dim
        self.token_idx = token_idx

    def provide_token_step(self, obs, info, token):
        pass

    def provide_token_reset(self, obs, info, token):
        self.provide_token_step(obs, info, token)
