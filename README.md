# MinecraftRL

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fyhs0602%2FMinecraftRL)](https://github.com/yhs0602/MinecraftRL)

RL experiments using lightweight minecraft environment

# Environment

https://github.com/KYHSGeekCode/MinecraftEnv

# Experiments

Please see [final_experiments](https://github.com/yhs0602/MinecraftRL/tree/main/code/experiments) to see various
settings for the experiments.

# Tasks

- Escape a husk in a superflat world using sound information
- Escape three husks in a superflat world using sound information
- Escape a warden in a superflat world using sound information
- Escape a husk in a superflat world using visual information
- Escape three husks in a superflat world using visual information
- Fish in a normal world
- Find a village in a normal world using flying

# Models

- DQN, Double DQN, Dueling DQN
    - CNN (stride = 2, kernel_size=5)
    - Fully Connected Network (hidden_dim = 128)
- Dueling DRQN
- A2C
- TODO: PPO, SAC, DDPG etc

# Encoding

- Vision: 3 channels, w, h rgb array
- Audio: `[Dx, Dz, [Dy]] * number of sounds, player hit sound, cos(yaw), sin(yaw)`

# Example Model Figures (DQN)

![Model architecture](./poster/models.png)

# Experiments

## How to make an environment

First, there are some concepts you need to know to create a custom experiment.

1. Environment: The environment that the agent interacts with. You may choose one
   from [environments.py](https://github.com/yhs0602/MinecraftRL/blob/b32c121e68ccdc4b262f64f8f74a1b46e29a0091/code/environments.py#L1103).
   The environment need a name and params to be created.
2. Wrapper: The wrapper helps you to connect your models, algorithms to the environment. It provides custom rewards, or
   observation space conversion.
3. Algorithm: It runs the training and test loops, and log the results. As the algorithm is tightly coupled with the
   model, you may need to implement your own algorithm to use your custom model.

To create a custom experiment, you need to create a new file in `code/experiments` directory. The file name should be
`{your_experiment_name}.yml`. Then, you need to specify the following specs. The specific specs may vary depending on
the algorithm you use.

```yaml
seed: null # The seed to generate the world. If null, the seed will be randomly generated.
env_path: null # The path to the MinecraftEnv project. If null, it will use the default path.
group: jsrl_fish # The name of the group of the experiment. The experiments will be grouped by this name in the wandb console.
record_video: true # Whether to record the video of the experiment.
device: null # The device to run the experiment. If null, it will use the default device.

env: # The environment to run the experiment in.
  name: # The name of the pre-defined environment. You can choose one from the list in environments.py
    params: # The parameters to create the environment. You can see the parameters in environments.py
      hud: true # Whether to show the HUD in the environment.
      verbose: false # Whether to show the verbose information in the environment.
      port: 8000 # Via which port the environment will communicate with this agent.
      render_action: true # Whether to include the agent's action in the video.
      size_x: 256 # The size of the screen in x axis.
      size_y: 256 # The size of the screen in y axis.

wrappers: # The wrappers that wraps the environment. It is an array of wrappers that will be applied in order.
  - name: 'ActionWrapper' # The convenience wrapper for the discrete actions.
    enabled_actions: # The specific actions that are enabled by this wrapper.
      - NO_OP # No operation action.
      - USE # Use action.

  - name: 'SoundWrapper' # The wrapper that adds sound-based features to the environment.
    coord_dim: 2 # The dimension of sound coordinates.
  - name: 'FishCodWrapper' # Additional wrapper for fishing reward.

algorithm: # Details of the algorithm to be used.
  name: "SoundJSRLDQNAlgorithm" # The specific name of the algorithm.
  params: # Parameters for the algorithm.
    num_episodes: 1000 # Total number of episodes to run.
    warmup_episodes: 10 # Episodes before starting the main training.
    steps_per_episode: 400 # Steps to be taken per episode.
    test_frequency: 10 # Frequency of tests during the episodes.
    solved_criterion: # Criterion to determine if the environment is solved.
      name: 'ScoreCriterion'
      params:
        min_episode: 100
        min_avg_score: 195
        min_test_score: 195
        min_avg_test_score: 195
  hidden_dim: 128 # Dimension of the hidden layers.
  epsilon_init: 1.0 # Initial value of epsilon for the epsilon-greedy policy.
  epsilon_decay: 0.99 # Decay rate of epsilon.
  epsilon_min: 0.01 # Minimum value of epsilon.
  update_frequency: 1000 # Frequency of updates.
  train_frequency: 1 # Training frequency.
  replay_buffer_size: 1000000 # Size of the replay buffer.
  batch_size: 256 # Batch size for training.
  gamma: 0.99 # Discount factor.
  learning_rate: 0.00001 # Learning rate for the optimizer.
  weight_decay: 0.00001 # Weight decay for regularization.
  tau: 1.0 # Target network update rate.
  guide_policy: # Policy that guides the agent.
    name: 'FishingGuide'
    params:
      min_episode: 100
  decrease_guide_step_threshold: 0.5 # Threshold for decreasing guide steps.
```

# Environment list

```
env_makers = {
    "husk": make_husk_environment,
    "husks": make_husks_environment,
    "husk-noisy": make_husk_noisy_environment,
    "husks-noisy": make_husks_noisy_environment,
    "husk-darkness": make_husk_darkness_environment,
    "husks-darkness": make_husks_darkness_environment,
    "find-animal": make_find_animal_environment,
    "husk-random": make_random_husk_environment,
    "husks-random": make_random_husks_environment,
    "husks-random-darkness": make_random_husks_darkness_environment,
    "husks-continuous": make_continuous_husks_environment,
    "husk-random-terrain": make_random_husk_terrain_environment,
    "husk-random-forest": make_random_husk_forest_environment,
    "husk-hunt": make_hunt_husk_environment,
    "mansion": make_mansion_environment,
    "skeleton-random": make_skeleton_random_environment,
    "find-village": make_find_village_environment,
    "flat-night": make_flat_night_environment,
    "fishing": make_fishing_environment,
}
```

| Env name              | Description                                                                   |
|-----------------------|-------------------------------------------------------------------------------|
| husk                  | Escaping from a single husk in a superflat world. The husk position is fixed. |
| husks                 | Escaping from multiple husks in a superflat world. The positions are fixed.   |
| husk-noisy            | Escaping from a husk, with many other animals.                                |
| husks-noisy           | Escaping from husks, with many other animals                                  |
| husk-darkness         | Escaping from a husk, with darkness effect                                    |
| husks-darkness        | Escaping from husks, with darkness effect                                     |
| find-animal           | Searching for randomly arranged animals in a animal pen                       |
| husk-random           | Escaping from a randomly positioned husk.                                     |
| husks-random          | Escaping from randomly positioned husks.                                      |
| husks-random-darkness | Escaping from randomly positioned husks with darkness effect applied          |
| husks-continuous      | Husks are summoned nearby the player continuously                             |
| husk-random-terrain   | Escape from a husk, in a normal terrain                                       |
| husk-random-forest    | Escape from a husk, in a forest                                               |
| husk-hunt             | Hunting a husk in a superflat world using a diamond sword.                    |
| mansion               | Escaping from a mansion                                                       |
| skeleton-random       | Escaping from a skeleton                                                      |
| find-village          | Searching for a village                                                       |
| flat-night            | Escaping from every threats in a superflat world at night                     |
| fishing               | Fish a cod on a beach                                                         |

# Wrapper list

| Wrapper Name            | Description                                                                                         |
|-------------------------|-----------------------------------------------------------------------------------------------------|
| CleanUpFastResetWrapper | A wrapper for fast environment resetting, every wrappers should inherit this.                       |
| action                  | Defines discrete action spaces and operations for the agent.                                        |
| continuous_action       | Enables the agent to take actions in a continuous action space.                                     |
| fly_helper              | Assists the agent in flying operations within the environment.                                      |
| mineclip                | Allows the agent to use [MineCLIP](https://github.com/MineDojo/MineCLIP/tree/main/mineclip) reward. |
| surrounding_sound       | Adds auditory feedback from the environment, indicating surrounding entities.                       |
| attack_kill             | Enables the agent to execute attack and eliminate operations.                                       |
| avoid_damage            | Helps the agent in strategies to prevent or minimize damage.                                        |
| bimodal                 | Provides vision and audio input for the agent.                                                      |
| find_animal             | Assists the agent in locating specific animals in the environment.                                  |
| find_village            | Aids the agent in discovering villages within the environment.                                      |
| fish_cod                | Enables the agent to perform fishing operations.                                                    |
| go_up                   | Assists the agent in upward movement or climbing actions.                                           |
| go_up_2                 | An extended version or variant of `go_up`, offering more functionalities.                           |
| jump_helper             | Aids the agent in performing jumping actions correctly.                                             |
| simple_navigation       | Provides basic navigation functionalities for the agent.                                            |
| simplest_navigation     | A more streamlined version of `simple_navigation` with minimal functionalities.                     |
| sound                   | Provides sound-based feedback or actions for the agent.                                             |
| survival                | Enables survival strategies and behaviors for the agent.                                            |
| terminate_on_death      | Ends the episode or session upon the agent's death.                                                 |
| vision                  | Incorporates visual feedback or vision-based actions for the agent.                                 |

# Algorithm list

| Algorithm Name | Description                                                                              |
|----------------|------------------------------------------------------------------------------------------|
| a2c            | Advantage Actor-Critic algorithm for policy and value function approximation.            |
| bimodal_dqn    | DQN variant designed for environments with bimodal observation space.                    |
| dqn            | Deep Q-Network algorithm for Q-value approximation using deep neural networks.           |
| epsilon_greedy | A simple exploration strategy using epsilon probability for random actions.              |
| jsrl_dqn       | Custom DQN variant specifically tailored for JSRL environments.                          |
| sound_a2c      | A2C algorithm with sound-based inputs or feedback.                                       |
| sound_dqn      | DQN variant that utilizes sound-based observations.                                      |
| sound_drqn     | Deep Recurrent Q-Network with sound inputs for environments with temporal dependencies.  |
| sound_jsrl_dqn | Custom JSRL DQN variant leveraging sound-based observations.                             |
| vision_a2c     | A2C algorithm with visual-based inputs or feedback.                                      |
| vision_dqn     | DQN variant that utilizes visual observations.                                           |
| vision_drqn    | Deep Recurrent Q-Network with visual inputs for environments with temporal dependencies. |
| bimodal_a2c    | A2C variant designed for environments with bimodal observations.                         |
| bimodal_drqn   | DRQN variant for environments with bimodal observations.                                 |
| drqn           | Deep Recurrent Q-Network for environments with temporal dependencies.                    |

# Models

| Model Name                    | Description                                                                                      |
|-------------------------------|--------------------------------------------------------------------------------------------------|
| dqn                           | Basic Deep Q-Network model for value function approximation.                                     |
| dueling_bimodal_attention_dqn | Dueling DQN with attention mechanism for bimodal inputs.                                         |
| dueling_bimodal_dqn           | Dueling DQN architecture for environments with bimodal observations.                             |
| dueling_sound_dqn             | Dueling DQN model that utilizes sound-based observations.                                        |
| dueling_sound_drqn            | Dueling Deep Recurrent Q-Network with sound inputs for environments with temporal dependencies.  |
| dueling_vision_dqn            | Dueling DQN model that utilizes visual observations.                                             |
| dueling_vision_drqn           | Dueling Deep Recurrent Q-Network with visual inputs for environments with temporal dependencies. |
| per                           | Prioritized Experience Replay mechanism to weigh experiences based on their TD-error.            |
| ppo                           | Proximal Policy Optimization, a policy gradient method for reinforcement learning.               |
| recurrent_replay_buffer       | Replay buffer designed for recurrent models to store sequences of experiences.                   |
| replay_buffer                 | Basic replay buffer to store and sample experiences.                                             |
| sound_a2c                     | Advantage Actor-Critic model tailored for sound-based observations.                              |
| vision_a2c                    | Advantage Actor-Critic model tailored for visual observations.                                   |
| bimodal_a2c                   | A2C model designed for environments with bimodal observations.                                   |
| bimodal_replay_buffer         | Replay buffer tailored for environments with bimodal observations.                               |
| sumtree                       | Data structure for efficient computation in prioritized experience replay.                       |
| transition                    | Data structure or method for representing state transitions in the environment.                  |

# Devaju font license

https://dejavu-fonts.github.io/License.html


