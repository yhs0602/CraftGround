# MinecraftRL

RL experiments using lightweight minecraft environment

# Environment

https://github.com/KYHSGeekCode/MinecraftEnv

# Experiments

Please see [final_experiments](https://github.com/KYHSGeekCode/MinecraftRL/tree/main/final_experiments) to see working
codes.

# Tasks

- Escape a husk in a superflat world using sound information
- Escape three husks in a superflat world using sound information
- Escape a warden in a superflat world using sound information
- Escape a husk in a superflat world using visual information
- Escape three husks in a superflat world using visual information
- Fish in a normal world

# Models

- DQN, Double DQN, Dueling DQN
    - CNN (stride = 2, kernel_size=5)
    - Fully Connected Network (hidden_dim = 128)
- TODO: Policy Gradient Methods

# Encoding

- CNN: 3 channels, w, h rgb array
- Audio: `[Dx, Dz, [Dy]] * number of sounds, player hit sound, cos(yaw), sin(yaw)`

# Models
![Model architecture](./poster/models.png)
