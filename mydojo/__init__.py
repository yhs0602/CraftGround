from initial_environment import InitialEnvironment
from mydojo.MyEnv import MyEnv


def make(**kwargs) -> MyEnv:
    env = InitialEnvironment(**kwargs)
    return MyEnv(env)
