# CraftGround Internals
## Capturing Frames
### Method 1: Using Default Screenshot Functionality
```java
public static NativeImage takeScreenshot(Framebuffer framebuffer)
```
  This method requires a `Framebuffer` object from the `MinecraftClient` class and is used to capture the current frame of the game. Eventually, the screen data must be converted into a `ByteString` object from protobuf.
### Method 2: Using `glGetTexImage` Function
The limitation of Method 1 involves multiple steps: reading pixel data from the texture, packing the alpha data, vertically mirroring, calling the `getBytes` method of `NativeImage`, reading it using `ImageIO`, resizing, writing into a `ByteArrayOutputStream`, converting to `ByteArray` (which copies the data again), and calling `ByteString.copyFrom()` (which also copies the data). To streamline this, we directly use the `glGetTexImage` function in native code, which reads pixel data from the texture and converts it directly into a `ByteString` object. This approach is faster as it minimizes data copying.
### Stage 3: Using `glReadPixels` Function (Current Method)
Although fast, Stage 2's method had a potential flaw: when calling the `glGetTexImage` function, rendering might not be completed, leading to outdated data capture. Thus, `glFinish` was necessary to ensure rendering completion, which could slow down the process. The current method utilizes the `glReadPixels` function, which inherently waits for rendering completion. This method is potentially quicker than combining `glGetTexImage` and `glFinish`, as it only waits for the necessary GL operations for the current frame rendering.
## Synchronizing Simulation
Minecraft uses two threads: one for `MinecraftClient`, which renders the game, and another for `MinecraftServer`, handling game logic. To synchronize, ensuring the client thread waits for the agent's action decision for the current tick and the server thread waits for the client's rendering completion, we employ the `TickSynchronizer` class. This synchronization allows seamless operation between client rendering, observation sending, and action reading from the agent.
## Offscreen Rendering
A major concern with 3D rendering environments is their functionality on GPU servers without a connected display, potentially leading to crashes or forced software rendering. By using [Xvfb](https://www.x.org/releases/X11R7.6/doc/man/man1/Xvfb.1.xhtml) and [VirtualGL](https://virtualgl.org/) for virtual display and GPU utilization, we ensure the environment runs on headless servers without issues, optimizing performance.
## Communication between Java and Python
To efficiently handle binary (e.g., screen captures) and text data (e.g., sounds, agent states), we use [protobuf](https://protobuf.dev/) for serialization and deserialization. Initially, one might consider base64 encoding within a JSON string, but this method is less efficient due to a 33% increase in data size. Protobuf adds minimal overhead, enhancing efficiency. For faster communication than TCP sockets, we utilize Unix domain sockets, ideal for the frequent data exchanges required every tick. Additionally, we send raw image data to avoid the slowdown from encoding/decoding processes, leaving data manipulation to the agent.

# Optimizations
To further improve the performance of the agent, we have implemented the following optimizations:
- **Optimization 1**: Using `glReadPixels` over `glGetTexImage` for data capture.
- **Optimization 2**: Incorporating [Sodium](https://github.com/CaffeineMC/sodium-fabric) and [Lithium](https://github.com/CaffeineMC/lithium-fabric) mods for rendering and simulation logic optimization.
- **Optimization 3**: Skipping world data saving to disk, unnecessary for agent learning. Traditionally happening every 6000 ticks (or five real-time minutes), this can be accelerated in simulations to occur every minute at 100 ticks per second (tps).
- **Optimization 4**: Omitting vertical image flipping in favor of numpy indexing on the Python side, optimizing channel swapping directly in the agent processing.
- **Optimization 5**: Using Unix domain sockets for communication between Java and Python, reducing latency and improving data exchange efficiency.
- **Optimization 6**: Adjusting JVM options for better performance.

# For contributors
## Fabric Mod
- Refer to the [Fabric Wiki](https://fabricmc.net/wiki/start) for mod development.

## Setting up the Development Environment
The overall project structure is as follows:
```tree
 tree -d -I venv -I build -I __pycache__ 
.
├── craftground
│   ├── MinecraftEnv (Git submodule: https://github.com/yhs0602/MinecraftEnv)
│   │   ├── gradle
│   │   │   └── wrapper
│   │   └── src
│   │       └── main
│   │           ├── cpp
│   │           ├── java
│   │           │   └── com
│   │           │       └── kyhsgeekcode
│   │           │           └── minecraft_env
│   │           │               ├── client
│   │           │               ├── mixin
│   │           │               └── proto
│   │           ├── proto
│   │           └── resources
│   ├── craftground
│   │   ├── proto
│   │   └── proto_proto
│   ├── environments
│   └── wrappers
├── dejavu-fonts-ttf-2.37
│   ├── fontconfig
│   └── ttf
├── docs
├── figures
├── paper
└── poster
    └── _minted-poster

```
Here the `craftground` directory contains the main project code, with the `MinecraftEnv` directory as a submodule for the Java part. The `dejavu-fonts-ttf-2.37` directory contains the DejaVu fonts for rendering actions in the video. The `docs`, `figures`, `paper`, and `poster` directories contain the documentation, figures, paper, and poster files, respectively.

Currently, you have to make pull requests to two repositories: [CraftGround](https://github.com/yhs0602/Craftground), and [MinecraftEnv](https://github.com/yhs0602/MinecraftEnv), which is a submodule of CraftGround. If you have better idea of managing this, please let us know via GitHub issue.


### Setting Up Building Native Code
- This project uses [Java Native Interface](https://docs.oracle.com/javase/8/docs/technotes/guides/jni/) for native code. Ensure you have the necessary tools installed.
- [CMake](https://cmake.org/) build system is used for the native code. Install CMake and ensure it is in your system's PATH.
- As `glBindFramebuffer` is used, the project requires [GLEW](https://glew.sourceforge.net/) (OpenGL Extension Wrangler Library) for OpenGL function loading. Ensure GLEW is installed on your system. Also, you need to call `glewInit()` before using any OpenGL functions. This is actually implemented in the native code. Though JVM side is already using OpenGL 3.0 functions and above, it is necessary to initialize GLEW for the native code, separately. 

## Adding New Observations
To add new observations, there are some steps to follow:
1. Add the new observation to the `ObservationSpaceMessage` in `observation_space.proto`.
2. run `protoc --python_out=../proto observation_space.proto` at `craftground/proto_proto` to generate the Python code.
3. Apply the same changes to the Java side. `src/main/proto/` contains the proto files for the Java side. You can run `protoc --kotlin_out=../java --java_out=../java observation_space.proto` at `MinecraftEnv/src/main/proto` to generate the Java and Kotlin code.
4. Now add logics to the Java/Kotlin side to capture the new observation. You can refer to the existing code in `MinecraftEnv/src/main/java/com/kyhsgeekcode/minecraft_env/Minecraft_env.kt` for adding various observations.
5. Add logics for converting received data to appropriate gymnasium observation data in the Python side. You can refer to the existing code in `craftground/craftground/craftground.py`'s `convert_observation` method.