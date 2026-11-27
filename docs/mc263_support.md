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

Sockets are the default transport and are required on Windows. On macOS, pass `use_shared_memory=True` to `craftground.make` to exchange actions and observations through POSIX shared memory. Install both the core and the 26.3 runtime from this branch. On Apple Silicon, the integration tests cover RAW, stereo PNG with depth, Metal tensor capture, fast resets, and full Minecraft restarts over shared memory.

## Capture modes

On Apple Silicon and Linux with an NVIDIA RTX 6000 Ada, the integration test exercises single-eye RAW and stereo PNG with depth on both OpenGL and Vulkan. It checks player displacement, a camera turn and image change in the same step, distinct stereo images, and depth at the requested resolution. Select the backend when launching Python with `CRAFTGROUND_MC263_GRAPHICS_BACKEND=OPENGL` or `VULKAN`.

The same suite targets a poplar plank, breaks it with a Python action, and confirms by raycast that the wall behind it is now visible. It checks the change in depth against the known distances to both surfaces and uses four colors on the wall to check RAW and PNG orientation at 319 × 181.

Apple Silicon also supports `ScreenEncodingMode.ZEROCOPY_TORCH` on Vulkan. Set `CRAFTGROUND_MC263_GRAPHICS_BACKEND=VULKAN` and add `-Dcraftground.enableMetalObjects=true` to `JAVA_TOOL_OPTIONS`. macOS may ask for administrator authorization when Minecraft shares its IOSurface with Python. The live test checks colors, orientation, and a camera turn on the returned PyTorch MPS tensor. With JAX installed, select `ScreenEncodingMode.ZEROCOPY_JAX` to capture through the same IOSurface. Python copies the frame to CPU if JAX cannot import the MPS tensor.

On Linux with NVIDIA, build the core Python extension and the 26.3 runtime with the CUDA toolkit available to CMake, and install a CUDA-enabled PyTorch. Select `ScreenEncodingMode.ZEROCOPY_TORCH`. OpenGL shares its color texture through CUDA. For Vulkan, add `-Dcraftground.enableCudaInterop=true` to `JAVA_TOOL_OPTIONS` before starting Python. [26.2 GPU interop design](26_2_vulkan_capture.md)

On Ubuntu 24.04 with an RTX 6000 Ada and driver 580.196.07, both backends returned CUDA tensors with the expected colors, orientation, and response to a camera turn. The test checks depth before and after breaking a block, then repeats the scene after restarting Minecraft in the same Python environment.

To run the installed-runtime checks from a checkout with pytest installed, use

```sh
CRAFTGROUND_MC263_GRAPHICS_BACKEND=OPENGL python -m pytest tests/python/native_integration/mc263_smoke.py
CRAFTGROUND_MC263_GRAPHICS_BACKEND=VULKAN python -m pytest tests/python/native_integration/mc263_smoke.py
```

To include the live Metal test on macOS, run

```sh
CRAFTGROUND_TEST_METAL=1 python -m pytest tests/python/native_integration/mc263_smoke.py -k metal
CRAFTGROUND_TEST_JAX=1 python -m pytest tests/python/native_integration/mc263_smoke.py -k jax
```

To run the live CUDA test on a Linux NVIDIA host with a working graphics display, use

```sh
CRAFTGROUND_TEST_CUDA=1 CRAFTGROUND_MC263_GRAPHICS_BACKEND=OPENGL python -m pytest tests/python/native_integration/mc263_smoke.py -k cuda
CRAFTGROUND_TEST_CUDA=1 CRAFTGROUND_MC263_GRAPHICS_BACKEND=VULKAN python -m pytest tests/python/native_integration/mc263_smoke.py -k cuda
```
