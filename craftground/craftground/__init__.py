from typing import Union

from .craftground import CraftGroundEnvironment, MultiDiscreteEnv
from .initial_environment import InitialEnvironment


def make(
    multidiscrete=False,
    verbose=False,
    env_path=None,
    port=8000,
    render_action=False,
    **kwargs,
) -> Union[CraftGroundEnvironment, MultiDiscreteEnv]:
    env = InitialEnvironment(**kwargs)
    if multidiscrete:
        return MultiDiscreteEnv(env)
    return CraftGroundEnvironment(
        env, verbose=verbose, env_path=env_path, port=port, render_action=render_action
    )
