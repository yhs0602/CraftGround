from typing import Union

from mydojo.initial_environment import InitialEnvironment
from mydojo.MyEnv import MyEnv, MultiDiscreteEnv


def make(
    multidiscrete=False, verbose=False, env_path=None, port=8000, **kwargs
) -> Union[MyEnv, MultiDiscreteEnv]:
    env = InitialEnvironment(**kwargs)
    if multidiscrete:
        return MultiDiscreteEnv(env)
    return MyEnv(env, verbose=verbose, env_path=env_path, port=port)
