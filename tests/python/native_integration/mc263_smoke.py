"""Run explicitly with pytest on a machine that can launch the Minecraft client."""

import math

import numpy as np
import pytest

import craftground
from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
from craftground.initial_environment_config import Difficulty, WorldType
from craftground.screen_encoding_modes import ScreenEncodingMode


@pytest.mark.parametrize(
    "mode,stereo", [(ScreenEncodingMode.RAW, False), (ScreenEncodingMode.PNG, True)]
)
def test_installed_runtime(mode, stereo):
    width, height = 320, 240
    config = craftground.InitialEnvironmentConfig(
        image_width=width,
        image_height=height,
        seed="43",
        world_type=WorldType.SUPERFLAT,
        difficulty=Difficulty.PEACEFUL,
        initial_extra_commands=["difficulty peaceful"],
        render_distance=3,
        hud_hidden=True,
        requires_depth=True,
        eye_distance=0.15 if stereo else 0.0,
        screen_encoding_mode=mode,
    )
    env = craftground.make(
        mc_version="26.3",
        initial_env_config=config,
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
    )
    try:
        obs, _ = env.reset()
        # Let spawn chunks reach the renderer before checking scene geometry.
        for _ in range(100):
            obs = env.step(no_op_v2())[0]
            if len(obs["full"].depth) and np.ptp(obs["full"].depth) > 0:
                break

        start = obs["full"]
        frame = obs["rgb"].copy()
        obs = env.step(dict(no_op_v2(), camera_yaw=30))[0]
        yaw_change = (obs["full"].yaw - start.yaw + 180) % 360 - 180
        assert yaw_change == pytest.approx(30, abs=2)
        assert np.mean(frame != obs["rgb"]) > 0.005
        if stereo:
            assert obs["rgb"].shape == obs["rgb_2"].shape
            assert np.any(obs["rgb"] != obs["rgb_2"])
        else:
            assert obs["rgb"].shape == (height, width, 3)

        for _ in range(12):
            obs = env.step(dict(no_op_v2(), forward=True))[0]
        end = obs["full"]
        assert math.hypot(end.x - start.x, end.z - start.z) > 0.3
        depth = np.asarray(end.depth)
        assert depth.size == width * height
        assert np.isfinite(depth).all() and np.ptp(depth) > 0

        reset_obs, _ = env.reset()
        assert reset_obs["rgb"].shape == frame.shape
        assert env.step(no_op_v2())[0]["rgb"].shape == frame.shape
    finally:
        env.close()
