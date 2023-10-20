from .craftground import CraftGroundEnvironment
from .initial_environment import InitialEnvironment


def make(
    verbose=False,
    env_path=None,
    port=8000,
    render_action=False,
    **kwargs,
) -> CraftGroundEnvironment:
    env = InitialEnvironment(**kwargs)
    return CraftGroundEnvironment(
        env, verbose=verbose, env_path=env_path, port=port, render_action=render_action
    )