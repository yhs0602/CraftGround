# Binocular Observation
CraftGround supports binocular observation, which provides a more realistic view of the environment. The binocular observation is enabled by setting the `biocular` parameter to `true` in the `craftground.make()` call. The binocular observation allows the agent to perceive the environment with a depth effect, similar to human vision.

The binocular vision observation is stored in `image_2` field, with the same shape as `image` but with a different perspective. The binocular observation can be used to improve the agent's perception of depth and distance in the environment.

To render alternating eyes, you have to pass `render_alternating_eyes=True` when calling `craftground.make()`. This will render the left and right eyes alternately, which helps checking the perception of the agent easily.  