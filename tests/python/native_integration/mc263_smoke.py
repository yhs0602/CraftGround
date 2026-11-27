"""Run explicitly with pytest on a machine that can launch the Minecraft client."""

import math
import os
import sys

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


@pytest.mark.parametrize(
    "mode,tensor_device",
    [
        (ScreenEncodingMode.RAW, None),
        (ScreenEncodingMode.PNG, None),
        pytest.param(
            ScreenEncodingMode.ZEROCOPY_TORCH,
            "mps",
            id="metal",
            marks=pytest.mark.skipif(
                sys.platform != "darwin" or os.environ.get("CRAFTGROUND_TEST_METAL") != "1",
                reason="Opt-in live Metal test requires macOS task-port authorization",
            ),
        ),
        pytest.param(
            ScreenEncodingMode.ZEROCOPY_TORCH,
            "cuda",
            id="cuda",
            marks=pytest.mark.skipif(
                sys.platform != "linux" or os.environ.get("CRAFTGROUND_TEST_CUDA") != "1",
                reason="Opt-in CUDA test requires a Linux NVIDIA GPU",
            ),
        ),
    ],
)
def test_block_interaction_and_frame_orientation(mode, tensor_device, monkeypatch):
    if tensor_device:
        backend = "VULKAN"
        if tensor_device == "cuda":
            backend = os.environ.get("CRAFTGROUND_MC263_GRAPHICS_BACKEND", backend).upper()
        monkeypatch.setenv("CRAFTGROUND_MC263_GRAPHICS_BACKEND", backend)
        if backend == "VULKAN":
            flag = "enableMetalObjects" if tensor_device == "mps" else "enableCudaInterop"
            options = os.environ.get("JAVA_TOOL_OPTIONS", "")
            monkeypatch.setenv(
                "JAVA_TOOL_OPTIONS", f"{options} -Dcraftground.{flag}=true".strip()
            )

    width, height = 319, 181
    config = craftground.InitialEnvironmentConfig(
        image_width=width,
        image_height=height,
        world_type=WorldType.SUPERFLAT,
        seed="43",
        render_distance=3,
        hud_hidden=True,
        request_raycast=True,
        requires_depth=True,
        screen_encoding_mode=mode,
        initial_extra_commands=[
            "difficulty peaceful",
            "time set noon",
            "weather clear",
            "gamemode creative @p",
            "fill -6 79 -6 6 79 6 smooth_stone",
            "tp @p 0.5 80 0.5 0 0",
            "fill 1 82 5 9 90 5 red_concrete",
            "fill -9 82 5 0 90 5 lime_concrete",
            "fill 1 73 5 9 81 5 blue_concrete",
            "fill -9 73 5 0 81 5 yellow_concrete",
            "setblock 0 81 3 poplar_planks",
        ],
    )
    env = craftground.make(
        mc_version="26.3",
        initial_env_config=config,
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
    )

    def target(obs):
        block = obs["full"].raycast_result.target_block
        return block.translation_key, block.x, block.y, block.z

    def depth_at(obs, x):
        depth = np.asarray(obs["full"].depth).reshape(height, width)
        return np.median(depth[88:93, x - 2 : x + 3])

    def check_scene():
        plank = ("block.minecraft.poplar_planks", 0, 81, 3)
        # Facing +Z at z=0.5, the plank face is 2.5 blocks away and the wall is 4.5.
        distance_ratio = 4.5 / 2.5
        for _ in range(100):
            obs = env.step(no_op_v2())[0]
            near = depth_at(obs, width // 2)
            expected_far = pytest.approx(near * distance_ratio, rel=0.03)
            if target(obs) == plank and near > 0 and depth_at(obs, 65) == expected_far:
                break
        else:
            pytest.fail(f"Plank or wall missing from raycast/depth: {target(obs)}, {near}")

        obs = env.step(dict(no_op_v2(), attack=True))[0]
        wall = ("block.minecraft.yellow_concrete", 0, 81, 5)
        for _ in range(100):
            if target(obs) == wall:
                break
            obs = env.step(no_op_v2())[0]
        assert target(obs) == wall

        # The raycast can see the wall before the chunk mesh is ready.
        for _ in range(100):
            obs = env.step(no_op_v2())[0]
            rgb = obs["rgb"]
            if tensor_device:
                assert rgb.device.type == tensor_device
                rgb = rgb.cpu().numpy()
            if mode == ScreenEncodingMode.PNG:
                rgb = rgb.transpose(2, 1, 0)  # PNG is channels, width, height.
            assert rgb.shape == (height, width, 3)
            red, green, blue, yellow = [
                np.median(rgb[y - 7 : y + 7, x - 7 : x + 7], axis=(0, 1))
                for y in (46, 132)
                for x in (65, 250)
            ]
            if (
                red[0] > 1.5 * max(red[1:])
                and green[1] > 1.25 * max(green[0], green[2])
                and blue[2] > 1.4 * max(blue[:2])
                and min(yellow[:2]) > 2 * yellow[2]
                and depth_at(obs, width // 2) == expected_far
            ):
                break
        else:
            pytest.fail(f"Unexpected quadrant colors: {red}, {green}, {blue}, {yellow}")

        if tensor_device:
            yaw = obs["full"].yaw
            obs = env.step(dict(no_op_v2(), camera_yaw=30))[0]
            assert (obs["full"].yaw - yaw + 180) % 360 - 180 == pytest.approx(30, abs=2)
            assert np.mean(rgb != obs["rgb"].cpu().numpy()) > 0.005

    try:
        env.reset()
        check_scene()
        if tensor_device == "cuda":
            env.reset(options={"fast_reset": False})
            check_scene()
    finally:
        env.close()
