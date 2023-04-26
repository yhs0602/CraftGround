from typing import Union

from mydojo.initial_environment import InitialEnvironment
from mydojo.MyEnv import MyEnv, MultiDiscreteEnv


def make(multidiscrete=False, **kwargs) -> Union[MyEnv, MultiDiscreteEnv]:
    env = InitialEnvironment(**kwargs)
    if multidiscrete:
        return MultiDiscreteEnv(env)
    return MyEnv(env)
