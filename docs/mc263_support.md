---
title: Minecraft 26.3
nav_order: 9
---

# Minecraft 26.3

CraftGround selects the 26.3 Fabric runtime with `mc_version="26.3"`. The default is still 1.21. The first reset builds and starts the client, downloads Minecraft assets, and creates a world.

## Installing from this branch

Use a full JDK 25 and set `JAVA_HOME` to that JDK so CMake can find the JNI headers. The runtime also needs CMake 3.28 or newer, a C++ toolchain, and OpenGL development libraries. Linux and Windows builds use GLEW. PNG capture requires libpng.

From a checkout with the Python native-extension build dependencies installed, run

```sh
python -m pip install ./minecraft/mc263 '.[mc263]'
```

For a macOS source build, the core Python extension also needs LibTorch. The release workflow uses LibTorch 2.5.1. With Python 3.11 and the matching PyTorch installed in the active environment, CMake can use its headers and libraries directly.

```sh
python -m pip install 'torch==2.5.1'
export CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DTorch_DIR=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')/Torch"
# Newer Apple Clang versions diagnose a specialization in the LibTorch 2.5.1 headers.
CXXFLAGS="-Wno-invalid-specialization" python -m pip install ./minecraft/mc263 '.[mc263]'
```

When installing locally built wheels, put both the core and runtime wheels in the same directory and use `python -m pip install --find-links ./dist 'craftground[mc263]==2.8.0'`.

## Running an episode

```python
import craftground
from craftground.environment.action_space import no_op

env = craftground.make(mc_version="26.3")
try:
    observation, info = env.reset()
    observation, reward, terminated, truncated, info = env.step(no_op())
    print(observation["rgb"].shape)
finally:
    env.close()
```

Set `cleanup_world=False` when creating the environment to keep a generated world after closing it. Opening an existing world by display name preserves its save.

Use the default socket transport on macOS. With `use_shared_memory=True`, the native observation writer crashed during the first reset on Apple Silicon.

## Capture modes

The Apple Silicon integration test exercises single-eye RAW and stereo PNG with depth on both OpenGL and Vulkan. It checks player displacement, a camera turn and image change in the same step, distinct stereo images, and depth at the requested resolution. Select the backend when launching Python with `CRAFTGROUND_MC263_GRAPHICS_BACKEND=OPENGL` or `VULKAN`.

The same suite targets a poplar plank, breaks it with a Python action, and confirms by raycast that the wall behind it is now visible. It checks the change in depth against the known distances to both surfaces and uses four colors on the wall to check RAW and PNG orientation at 319 × 181.

The 26.3 runtime also includes the opt-in Vulkan zero-copy implementations inherited from 26.2. On this port, Metal reached device and IOSurface initialization. End-to-end tensor verification requires permission for the macOS Mach-port transfer. The CUDA path awaits verification on an NVIDIA system. [26.2 GPU interop design](26_2_vulkan_capture.md)

To run the installed-runtime checks from a checkout with pytest installed, use

```sh
CRAFTGROUND_MC263_GRAPHICS_BACKEND=OPENGL python -m pytest tests/python/native_integration/mc263_smoke.py
CRAFTGROUND_MC263_GRAPHICS_BACKEND=VULKAN python -m pytest tests/python/native_integration/mc263_smoke.py
```
