# Bimodal Observation
CraftGround supports not only (binocular) vision observation, but also sound subtitles, which provides a bimodal observation experience. The sound subtitles are actually the Minecraft's translation keys of its sound subtitles option in the accessibility settings. It provides relative position information of the sound source. We recommend adding the agent's yaw value to the observation of the agent to make it handle the position information of the sound source more effectively.

This short snippet shows how you can use bimodal observations for your agent. 

```python

env, sound_list = craftground.make(...)
env = FastResetWrapper(
    ActionWrapper(
        BimodalWrapper(env, x_dim=114, y_dim=64, sound_list=sound_list),
        enabled_actions=[
            Action.NO_OP,
            Action.FORWARD,
            Action.BACKWARD,
            Action.STRAFE_LEFT,
            Action.STRAFE_RIGHT,
            Action.TURN_LEFT,
            Action.TURN_RIGHT,
        ],
    ),
)
```