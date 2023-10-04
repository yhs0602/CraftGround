import abc


class BaseEnvironment(abc.ABC):
    def make(
        self,
        verbose: bool,
        env_path: str,
        port: int,
        size_x: int = 114,
        size_y: int = 64,
        hud: bool = False,
        render_action: bool = True,
        *args,
        **kwargs,
    ):
        return self
