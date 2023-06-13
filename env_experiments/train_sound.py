import argparse
import subprocess

import numpy as np
import torch
from gymnasium.wrappers import FrameStack

from env_wrappers.husk_environment import env_makers
from env_wrappers.sound_wrapper import SoundWrapper

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.has_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

parser = argparse.ArgumentParser(description="Run experiment")

parser.add_argument(
    "--verbose",
    type=bool,
    help="Verbose",
    required=False,
    default=False,
)

parser.add_argument(
    "--env_path",
    type=str,
    help="Path to environment",
    required=False,
    default=None,
)

parser.add_argument(
    "--agent",
    type=str,
    help="Agent to use (DDQN, DQN)",
    required=True,
)

parser.add_argument(
    "--env",
    type=str,
    help="Environment to use",
    required=True,
)

parser.add_argument(
    "--batch_size",
    type=int,
    help="batch size to use",
    required=True,
)

parser.add_argument(
    "--gamma",
    type=float,
    help="gamma to use",
    required=True,
)

parser.add_argument(
    "--learning_rate",
    type=float,
    help="learning rate to use",
    required=True,
)

parser.add_argument(
    "--update_freq",
    type=int,
    help="update frequency to use",
    required=True,
)

parser.add_argument(
    "--hidden_dim",
    type=int,
    help="hidden dimension to use",
    required=False,
    default=128,
)

parser.add_argument(
    "--weight_decay",
    type=float,
    help="weight decay to use",
    required=True,
)

# parser.add_argument(
#     "--kernel_size",
#     type=int,
#     help="kernel size to use",
#     required=True,
# )

# parser.add_argument(
#     "--stride",
#     type=int,
#     help="stride to use",
#     required=True,
# )

parser.add_argument(
    "--buffer_size",
    type=int,
    help="buffer size to use",
    required=True,
)

parser.add_argument(
    "--epsilon_init",
    type=float,
    help="epsilon decay to use",
    required=True,
)

parser.add_argument(
    "--epsilon_decay",
    type=float,
    help="epsilon decay to use",
    required=True,
)

parser.add_argument(
    "--epsilon_min",
    type=float,
    help="epsilon min to use",
    required=True,
)

parser.add_argument(
    "--max_steps_per_episode",
    type=int,
    help="max steps per episode to use",
    required=True,
)

parser.add_argument(
    "--num_episodes",
    type=int,
    help="number of episodes to use",
    required=True,
)

parser.add_argument(
    "--warmup_episodes",
    type=int,
    help="number of warmup episodes to use",
    required=True,
)


def train_sound(
    verbose,
    env_path,
    port,
    agent,
    env_name,
    batch_size,
    gamma,
    learning_rate,
    update_freq,
    hidden_dim,
    weight_decay,
    buffer_size,
    epsilon_init,
    epsilon_decay,
    epsilon_min,
    max_steps_per_episode,
    num_episodes,
    warmup_episodes,
    reward_function=None,
    stack_size=None,
):
    env, sound_list = env_makers[env_name](verbose, env_path, port)
    wrapper = SoundWrapper(
        env,
        action_dim=7,
        sound_list=sound_list,
        coord_dim=2,
        reward_function=reward_function,
    )
    if stack_size is not None:
        wrapper = FrameStack(wrapper, stack_size)
    if agent == "DQNAgent":
        if stack_size is None:
            from models.dqn import DQNSoundAgent

            agent_class = DQNSoundAgent
        else:
            from models.stacked_dqn import StackedDQNSoundAgent

            agent_class = StackedDQNSoundAgent
    elif agent == "DDQNAgent":
        from models.dqn import DDQNSoundAgent

        agent_class = DDQNSoundAgent
    elif agent == "DuelingDQNAgent":

        from models.dueling_dqn import DuelingSoundDQNAgent

        agent_class = DuelingSoundDQNAgent
    else:
        print(f"Agent not implemented: {agent}")
        raise NotImplementedError
    state_dim = wrapper.observation_space.shape
    print(state_dim)
    state_dim = (np.prod(state_dim),)
    print(state_dim)
    action_dim = wrapper.action_space.n
    agent_instance = agent_class(
        state_dim,
        action_dim,
        hidden_dim,
        buffer_size,
        batch_size,
        gamma,
        learning_rate,
        weight_decay,
        stack_size=stack_size,
        device=device,
    )

    from wrapper_runners.dqn_wrapper_runner import DQNWrapperRunner

    print("Running DQN wrapper runner")
    runner = DQNWrapperRunner(
        wrapper,
        env_name=env_name,
        agent=agent_instance,
        max_steps_per_episode=max_steps_per_episode,
        num_episodes=num_episodes,
        test_frequency=20,
        solved_criterion=lambda avg_score, test_score, avg_test_score, episode: avg_score
        >= 195.0
        and avg_test_score >= 195.0
        and episode >= 1000
        and test_score == 200.0
        if avg_score is not None
        else False and episode >= 1000,
        after_wandb_init=lambda *args: None,
        warmup_episodes=warmup_episodes,
        update_frequency=update_freq,
        epsilon_init=epsilon_init,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        resume=False,
        max_saved_models=1,
    )
    runner.run_wrapper(record_video=True)


if __name__ == "__main__":
    args = parser.parse_args()
    # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # sock.bind(("", 0))
    # port = sock.getsockname()[1]
    # sock.close()
    port = 8008
    train_sound(
        args.verbose,
        args.env_path,
        port,
        args.agent,
        args.env,
        args.batch_size,
        args.gamma,
        args.learning_rate,
        args.update_freq,
        args.hidden_dim,
        args.weight_decay,
        args.buffer_size,
        args.epsilon_init,
        args.epsilon_decay,
        args.epsilon_min,
        args.max_steps_per_episode,
        args.num_episodes,
        args.warmup_episodes,
    )
    try:
        pass
        # cmd = f"lsof -i:{port} | grep java | grep CLOSED | awk '{{print $2}}'"
        # process = subprocess.Popen(
        #     cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        # )
        # output, _ = process.communicate()
        # pid = int(output.strip())
        # subprocess.call(["kill", "-9", str(pid)])
        # print(f"Process with PID {pid} has been terminated.")
    except subprocess.CalledProcessError:
        print(f"No process found listening on port {port}.")
