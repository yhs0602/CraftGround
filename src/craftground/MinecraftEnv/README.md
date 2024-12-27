# MinecraftEnv

Lightweight minecraft RL environment on fabric

# Features

- Works on minecraft 1.21
- Used fabric mod
- JDK: 21

# initial environment
- Inventory commands
- Entities
- World seed
- is world superflat
- is hardcore
- initial x, y, z
- always night / day
- do mob spawn
- screen size (visible/observed)
- Arbitrary commands (e.g. status effects)
- keys for statistics
- Hud is visible

# Observation space

- rgb
- x, y, z
- pitch, yaw
- health, food_level, saturation_level
- is_dead
- inventory
- looking at entites, blocks
- sound subtitles
- status effects
- statistics
- visible entities
- surrounding entities
- is bobber thrown

# Action space

- Similar to https://docs.minedojo.org/sections/core_api/action_space.html
- No crafting yet
- Arbitrary command (you should hide this for the agents)

Controlled by this reinforcement learning agent: https://github.com/yhs0602/CraftGround

renderDistance:6
simulationDistance:6


# Build cpp
```
mkdir build
cd build
cmake ../src/main/cpp
```