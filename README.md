# MinecraftRL
RL experiments using lightweight minecraft environment

# Environment
https://github.com/KYHSGeekCode/MinecraftEnv

# Tasks
- Escape a husk in a superflat world using sound information
- Escape three husks in a superflat world using sound information
- Escape a warden in a superflat world using sound information
- Escape a husk in a superflat world using visual information
- Escape three husks in a superflat world using visual information
- Fish in a normal world

# Models
- DQN
  - CNN (stride = 2, kernel_size=5)
  - Fully Connected Network (hidden_dim = 128)
- TODO: other models

# Encoding
- CNN: 3 channels, x, y
- Audio: sound subtitles encoded to a sequence of (dx, dy), -1 <= dx, dy <= 1

