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


TRANSPORTS = [
    pytest.param(False, id="socket"),
    pytest.param(
        True,
        id="shmem",
        marks=pytest.mark.skipif(os.name != "posix", reason="POSIX shared memory"),
    ),
]


@pytest.mark.parametrize(
    "mode,stereo", [(ScreenEncodingMode.RAW, False), (ScreenEncodingMode.PNG, True)]
)
@pytest.mark.parametrize("shared_memory", TRANSPORTS)
def test_installed_runtime(mode, stereo, shared_memory):
    width, height = 320, 240
    commands = ["difficulty peaceful"]
    if stereo:
        commands.extend(
            [
                "time set noon",
                "fill -6 79 -6 6 79 8 smooth_stone",
                "tp @p 0.5 80 0.5 0 0",
                "fill -6 80 6 6 85 6 white_concrete",
                "setblock 0 81 3 red_concrete",
            ]
        )
    config = craftground.InitialEnvironmentConfig(
        image_width=width,
        image_height=height,
        seed="43",
        world_type=WorldType.SUPERFLAT,
        difficulty=Difficulty.PEACEFUL,
        initial_extra_commands=commands,
        mined_stat_keys=["stone", "minecraft:stone"],
        misc_stat_keys=["jump", "minecraft:jump"],
        render_distance=3,
        hud_hidden=True,
        requires_depth=True,
        eye_distance=0.15 if stereo else 0.0,
        screen_encoding_mode=mode,
    )
    env = craftground.make(
        mc_version="26.3",
        use_shared_memory=shared_memory,
        initial_env_config=config,
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
    )
    try:
        obs, _ = env.reset()
        for stats, key in (
            (obs["full"].mined_statistics, "stone"),
            (obs["full"].misc_statistics, "jump"),
        ):
            assert set(stats) == {key, f"minecraft:{key}"}
            assert stats[key] == stats[f"minecraft:{key}"]
        # Let spawn chunks reach the renderer before checking scene geometry.
        for _ in range(100):
            obs = env.step(no_op_v2())[0]
            if len(obs["full"].depth) and np.ptp(obs["full"].depth) > 0:
                break

        if stereo:
            # At yaw zero the red block is centered ahead of the player. Both
            # edges of its face should appear farther right in the left eye.
            for _ in range(100):
                obs = env.step(no_op_v2())[0]
                edges = []
                for image in (obs["rgb"], obs["rgb_2"]):
                    red, green, blue = image[:, :, height // 2].astype(int)
                    xs = np.flatnonzero((red > 2 * green) & (red > 2 * blue))
                    if xs.size:
                        edges.append((xs[0], xs[-1]))
                if len(edges) == 2:
                    break
            assert len(edges) == 2, "Red block missing from stereo view"
            left, right = edges
            assert left[0] > right[0] and left[1] > right[1], (left, right)

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

        reset_obs, _ = env.reset(options={"extra_commands": ["tp @p 5.5 85 5.5 0 0"]})
        assert (reset_obs["full"].x, reset_obs["full"].z) == pytest.approx((5.5, 5.5))
        first = env.step(dict(no_op_v2(), camera_yaw=30))[0]["full"]
        assert first.yaw == pytest.approx(30, abs=2)
        assert reset_obs["rgb"].shape == frame.shape
        assert env.step(no_op_v2())[0]["rgb"].shape == frame.shape
        if shared_memory:
            # A command batch can exceed the space used by the initial config.
            env.add_commands(["time set noon"] * 128 + ["tp @p 10 80 10"])
            for _ in range(12):
                full = env.step(no_op_v2())[0]["full"]
                if abs(full.x - 10) < 1 and abs(full.z - 10) < 1:
                    break
            assert abs(full.x - 10) < 1 and abs(full.z - 10) < 1
            obs, _ = env.reset(options={"fast_reset": False})
            assert obs["rgb"].shape == frame.shape
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
            ScreenEncodingMode.ZEROCOPY_JAX,
            "mps" if sys.platform == "darwin" else "cuda",
            id="jax",
            marks=pytest.mark.skipif(
                os.environ.get("CRAFTGROUND_TEST_JAX") != "1",
                reason="Opt-in JAX test requires native GPU capture",
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
@pytest.mark.parametrize("shared_memory", TRANSPORTS)
def test_block_interaction_and_frame_orientation(
    mode, tensor_device, shared_memory, monkeypatch
):
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
        use_shared_memory=shared_memory,
        initial_env_config=config,
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
    )

    def target(obs):
        block = obs["full"].raycast_result.target_block
        return block.translation_key, block.x, block.y, block.z

    def depth_at(obs, x):
        depth = np.asarray(obs["full"].depth).reshape(height, width)
        return np.median(depth[88:93, x - 2 : x + 3])

    def rgb_pixels(obs):
        frame = obs["rgb"]
        if mode == ScreenEncodingMode.ZEROCOPY_JAX:
            import jax

            assert isinstance(frame, jax.Array)
            return np.asarray(frame)
        if tensor_device:
            assert frame.device.type == tensor_device
            return frame.cpu().numpy()
        return frame.transpose(2, 1, 0) if mode == ScreenEncodingMode.PNG else frame

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
            rgb = rgb_pixels(obs)
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
            assert np.mean(rgb != rgb_pixels(obs)) > 0.005

    try:
        env.reset()
        check_scene()
        if tensor_device:
            env.reset(options={"fast_reset": False})
            check_scene()
    finally:
        env.close()
