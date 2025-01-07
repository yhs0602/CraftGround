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

# Docs on MinecraftEnv.kt
## ResetPhase
1. Initial: END_RESET
2. After reading initial environment: WAIT_INIT_ENDS

## IOPhase
1. Initial: BEGINNING
2. After reading initial environment: GOT_INITIAL_ENVIRONMENT_SHOULD_SEND_OBSERVATION
3. 