import time

import numpy as np
import yaml

import models
from env_wrappers.husk_environment import env_makers
from final_experiments import wrappers
from final_experiments.runners.bimodal import train_vision_and_sound
from final_experiments.runners.sound import train_sound
from final_experiments.runners.vision import train_cnn


def run_experiment():
    with open("file.yaml", "r") as yaml_file:
        yaml_content = yaml_file.read()
    data = yaml.safe_load(yaml_content)
    print(data)

    seed = int(time.time())
    np.random.seed(seed)

    verbose = data["verbose"]  # False
    env_path = data["env_path"]  # None
    port = data["port"]  # 8002
    env_data = data["env"]
    env_name = env_data["name"]  # "husk-random-terrain"
    hud_hidden = env_data["hud_hidden"]  # True

    inner_env, sound_list = env_makers[env_name](
        verbose, env_path, port, hud_hidden=hud_hidden
    )

    data_wrappers = data["wrappers"]
    for wrapper_name in data_wrappers.keys():
        wrapper_class = getattr(wrappers, wrapper_name)
        wrapper_instance = wrapper_class(
            env=inner_env, sound_list=sound_list, **data_wrappers[wrapper_name]
        )
        inner_env = wrapper_instance

    env = inner_env
    mode = data["mode"]

    if mode == "sound":
        fn = train_sound
    elif mode == "vision":
        fn = train_cnn
    elif mode == "bimodal":
        fn = train_vision_and_sound
    else:
        raise ValueError(f"Unknown mode {mode}")

    agent_class_name = data["agent_class"]
    agent_class = getattr(models, agent_class_name)
    solved_criterion = data["solved_criterion"]
    fn(
        env=env,
        agent_class=agent_class,
        **data["train"],
        solved_criterion=solved_criterion,
    )


if __name__ == "__main__":
    run_experiment()
