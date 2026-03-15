# CraftGround 아키텍처 플로우차트

이 문서는 CraftGround 레포지토리의 구성요소와 데이터 흐름을 매우 자세히 나타낸 플로우차트입니다.

## 전체 시스템 아키텍처

```mermaid
graph TB
    subgraph "Python Layer (RL Agent)"
        A[User Code] -->|craftground.make()| B[CraftGroundEnvironment]
        B -->|Gymnasium API| C[RL Framework<br/>Stable-Baselines3/RLlib/etc]
    end
    
    subgraph "Python Environment Layer"
        B --> D[InitialEnvironmentConfig]
        B --> E[ActionSpaceVersion]
        B --> F[IPC Interface]
        F -->|Socket Mode| G[SocketIPC]
        F -->|Shared Memory Mode| H[BoostIPC]
        B --> I[ObservationConverter]
        B --> J[CsvLogger]
    end
    
    subgraph "IPC Communication Layer"
        G -->|Unix Domain Socket<br/>/tmp/minecraftrl_PORT.sock| K[DomainSocketMessageIO]
        G -->|TCP Socket<br/>127.0.0.1:PORT| K
        H -->|Shared Memory<br/>craftground_PORT_p2j/j2p| L[SharedMemoryMessageIO]
        H -->|C++ Native| M[ipc_boost.cpp<br/>ipc_cuda.cpp<br/>ipc_apple.mm]
    end
    
    subgraph "Protocol Buffers"
        N[action_space.proto] --> O[ActionSpaceMessageV2]
        P[observation_space.proto] --> Q[ObservationSpaceMessage]
        R[initial_environment.proto] --> S[InitialEnvironmentMessage]
    end
    
    subgraph "Minecraft Mod Layer (Java/Kotlin)"
        K --> T[MinecraftEnv Mod]
        L --> T
        T --> U[MessageIO Interface]
        T --> V[EnvironmentInitializer]
        T --> W[FramebufferCapturer]
        T --> X[ObservationCollector]
        T --> Y[ActionExecutor]
        T --> Z[TickSynchronizer]
    end
    
    subgraph "Minecraft Game Engine"
        T --> AA[Minecraft Client<br/>Fabric Mod Loader]
        AA --> BB[Minecraft 1.21.0]
        BB --> CC[World Rendering]
        BB --> DD[Entity Management]
        BB --> EE[Block System]
    end
    
    style A fill:#e1f5ff
    style B fill:#b3e5fc
    style F fill:#81d4fa
    style T fill:#4fc3f7
    style AA fill:#29b6f6
```

## 초기화 플로우

```mermaid
sequenceDiagram
    participant User
    participant Env as CraftGroundEnvironment
    participant IPC as IPC Interface
    participant Gradle as Gradle Process
    participant Mod as MinecraftEnv Mod
    participant MC as Minecraft Client
    
    User->>Env: craftground.make(config)
    Env->>Env: Initialize ActionSpace
    Env->>Env: Initialize ObservationSpace
    Env->>Env: Create InitialEnvironmentMessage
    
    alt use_shared_memory == True
        Env->>IPC: BoostIPC(port, config)
        IPC->>IPC: initialize_shared_memory()<br/>(C++ native)
        IPC->>IPC: Create shared memory segments<br/>p2j and j2p
    else use_shared_memory == False
        Env->>IPC: SocketIPC(port, config)
        IPC->>IPC: Check/find free port
        IPC->>IPC: Prepare socket path<br/>(Unix: /tmp/minecraftrl_PORT.sock<br/>Windows: TCP 127.0.0.1:PORT)
    end
    
    Env->>Gradle: subprocess.Popen(gradlew runClient)
    Gradle->>MC: Start Minecraft Client
    MC->>Mod: onInitialize()
    
    alt Shared Memory Mode
        Mod->>Mod: SharedMemoryMessageIO(port)
        Mod->>Mod: Connect to shared memory
    else Socket Mode
        Mod->>Mod: Create ServerSocketChannel
        Mod->>Mod: Accept connection from Python
        IPC->>Mod: Connect to socket
        Mod->>Mod: DomainSocketMessageIO(socket)
    end
    
    IPC->>Mod: Send InitialEnvironmentMessage
    Mod->>Mod: Read InitialEnvironmentMessage
    Mod->>Mod: EnvironmentInitializer.initialize()
    Mod->>MC: Configure world settings
    Mod->>MC: Setup initial environment
    Mod->>Mod: Collect first observation
    Mod->>IPC: Send ObservationSpaceMessage
    IPC->>Env: Read observation
    Env->>Env: ObservationConverter.convert()
    Env->>User: Return initial observation
```

## Step 액션 플로우

```mermaid
sequenceDiagram
    participant Agent as RL Agent
    participant Env as CraftGroundEnvironment
    participant IPC as IPC Interface
    participant Mod as MinecraftEnv Mod
    participant MC as Minecraft Client
    participant Conv as ObservationConverter
    
    Agent->>Env: env.step(action)
    Env->>Env: Translate action V1→V2<br/>(if needed)
    Env->>Env: action_v2_dict_to_message()
    Env->>Env: Create ActionSpaceMessageV2
    
    Env->>IPC: send_action(action_msg, commands)
    
    alt Socket Mode
        IPC->>IPC: Serialize protobuf message
        IPC->>IPC: Write length (4 bytes LE)
        IPC->>IPC: Write message bytes
        IPC->>Mod: Send via socket
    else Shared Memory Mode
        IPC->>IPC: Serialize protobuf message
        IPC->>IPC: write_to_shared_memory()<br/>(C++ native)
        IPC->>Mod: Write to p2j shared memory
    end
    
    Mod->>Mod: readAction()
    Mod->>Mod: Parse ActionSpaceMessageV2
    
    Mod->>Mod: ActionExecutor.execute()
    Mod->>MC: Apply movement (forward/back/left/right)
    Mod->>MC: Apply camera (pitch/yaw)
    Mod->>MC: Apply actions (jump/sneak/sprint/attack/use)
    Mod->>MC: Handle hotbar selection
    Mod->>MC: Execute commands (if any)
    
    MC->>MC: Update game state (1 tick)
    MC->>Mod: Render frame
    
    Mod->>Mod: ObservationCollector.collect()
    Mod->>Mod: Capture framebuffer (RGB)
    Mod->>Mod: Collect player state (position, yaw, pitch)
    Mod->>Mod: Collect inventory
    Mod->>Mod: Collect nearby entities
    Mod->>Mod: Collect block info
    Mod->>Mod: Collect biome info
    Mod->>Mod: Collect lidar data (if enabled)
    Mod->>Mod: Collect sound data (if enabled)
    
    Mod->>Mod: Build ObservationSpaceMessage
    Mod->>IPC: writeObservation(observation)
    
    alt Socket Mode
        IPC->>IPC: Serialize protobuf message
        IPC->>IPC: Write length + message
        IPC->>Env: Receive via socket
    else Shared Memory Mode
        IPC->>IPC: read_from_shared_memory()<br/>(C++ native)
        IPC->>Env: Read from j2p shared memory
    end
    
    Env->>Conv: convert(observation)
    
    alt Screen Encoding Mode
        alt PNG Mode
            Conv->>Conv: Decode PNG bytes
            Conv->>Conv: Convert to numpy array
        else RAW Mode
            Conv->>Conv: Decode raw bytes
            Conv->>Conv: Reshape to image array
        else ZeroCopy Mode (CUDA/Apple)
            Conv->>Conv: Create tensor from memory handle
            Conv->>Conv: Return torch/jax tensor
        else JAX Mode
            Conv->>Conv: Convert to jax.numpy array
        end
    end
    
    Conv->>Env: Return RGB images (1 or 2 for binocular)
    Env->>Env: Build TypedObservation dict
    Env->>Agent: Return (obs, reward, done, truncated, info)
```

## Reset 플로우

```mermaid
sequenceDiagram
    participant Agent as RL Agent
    participant Env as CraftGroundEnvironment
    participant IPC as IPC Interface
    participant Mod as MinecraftEnv Mod
    participant MC as Minecraft Client
    
    Agent->>Env: env.reset(seed=seed, options=options)
    
    Env->>Env: ensure_alive(fast_reset, extra_commands, seed)
    
    alt Server is alive AND fast_reset == True
        Env->>IPC: send_fastreset2(extra_commands)
        IPC->>Mod: Send fast reset command
        Mod->>Mod: ResetPhase = WAIT_PLAYER_DEATH
        Mod->>MC: Kill player
        Mod->>Mod: Wait for respawn
        Mod->>Mod: Execute extra_commands
        Mod->>Mod: ResetPhase = END_RESET
    else Server is NOT alive OR fast_reset == False
        Env->>Env: terminate() (if alive)
        Env->>Env: start_server(seed)
        Env->>Gradle: Launch new Minecraft process
        Note over Gradle,MC: Full initialization sequence
    end
    
    Mod->>Mod: Collect observation after reset
    Mod->>IPC: Send ObservationSpaceMessage
    IPC->>Env: Read observation
    Env->>Env: Convert observation
    Env->>Agent: Return initial observation
```

## IPC 통신 상세 구조

```mermaid
graph LR
    subgraph "Python Side"
        A[SocketIPC/BoostIPC] -->|Serialize| B[Protobuf Message]
        B -->|ActionSpaceMessageV2| C[Binary Data]
    end
    
    subgraph "Transport Layer"
        C -->|Socket Mode| D[Unix Domain Socket<br/>or TCP Socket]
        C -->|Shared Memory Mode| E[Shared Memory<br/>p2j: Python→Java<br/>j2p: Java→Python]
    end
    
    subgraph "Java/Kotlin Side"
        D --> F[DomainSocketMessageIO]
        E --> G[SharedMemoryMessageIO]
        F --> H[Parse Protobuf]
        G --> H
        H --> I[ActionSpaceMessageV2]
    end
    
    subgraph "Observation Flow"
        J[ObservationSpaceMessage] -->|Serialize| K[Binary Data]
        K -->|Socket Mode| D
        K -->|Shared Memory Mode| E
        D --> F
        E --> G
        F --> L[Parse Protobuf]
        G --> L
        L --> M[ObservationSpaceMessage]
    end
    
    style A fill:#81d4fa
    style F fill:#4fc3f7
    style G fill:#4fc3f7
    style E fill:#ffcc80
```

## 관찰 공간 (Observation Space) 구조

```mermaid
graph TB
    A[ObservationSpaceMessage] --> B[Image Data]
    A --> C[Player State]
    A --> D[Inventory]
    A --> E[Entities]
    A --> F[Blocks]
    A --> G[Biomes]
    A --> H[Lidar]
    A --> I[Sound]
    
    B --> B1[image: bytes<br/>PNG or RAW]
    B --> B2[image_2: bytes<br/>Binocular mode]
    B --> B3[depth: bytes<br/>If requiresDepth]
    
    C --> C1[x, y, z: float<br/>Position]
    C --> C2[yaw, pitch: float<br/>Camera angles]
    C --> C3[health, food: float]
    C --> C4[isAlive: bool]
    
    D --> D1[inventory: repeated Item]
    D --> D2[mainHand: int]
    
    E --> E1[entitiesWithinDistance:<br/>repeated EntityInfo]
    
    F --> F1[blockInfo:<br/>repeated BlockInfo]
    F --> F2[heightInfo:<br/>repeated HeightInfo]
    
    G --> G1[biomeInfo:<br/>repeated BiomeInfo]
    G --> G2[nearbyBiome:<br/>repeated NearbyBiome]
    
    H --> H1[lidarResult:<br/>LidarResult]
    
    I --> I1[soundData: bytes]
    
    style A fill:#e1f5ff
    style B fill:#b3e5fc
    style C fill:#81d4fa
```

## 액션 공간 (Action Space) 구조

```mermaid
graph TB
    A[ActionSpaceMessageV2] --> B[Boolean Actions]
    A --> C[Continuous Actions]
    A --> D[Commands]
    
    B --> B1[forward: bool]
    B --> B2[back: bool]
    B --> B3[left: bool]
    B --> B4[right: bool]
    B --> B5[jump: bool]
    B --> B6[sneak: bool]
    B --> B7[sprint: bool]
    B --> B8[attack: bool]
    B --> B9[use: bool]
    B --> B10[drop: bool]
    B --> B11[inventory: bool]
    B --> B12[hotbar_1-9: bool]
    
    C --> C1[camera_pitch: float<br/>-180 to 180 degrees]
    C --> C2[camera_yaw: float<br/>-180 to 180 degrees]
    
    D --> D1[commands: repeated string<br/>Minecraft commands]
    
    style A fill:#e1f5ff
    style B fill:#c8e6c9
    style C fill:#fff9c4
    style D fill:#ffccbc
```

## 환경 초기화 설정 구조

```mermaid
graph TB
    A[InitialEnvironmentConfig] --> B[World Settings]
    A --> C[Image Settings]
    A --> D[Player Settings]
    A --> E[Feature Settings]
    A --> F[Structure Settings]
    
    B --> B1[isWorldFlat: bool]
    B --> B2[worldType: string]
    B --> B3[seed: int]
    B --> B4[timeOfDay: int]
    B --> B5[weather: string]
    
    C --> C1[imageSizeX: int]
    C --> C2[imageSizeY: int]
    C --> C3[screen_encoding_mode:<br/>PNG/RAW/ZEROCOPY/JAX]
    C --> C4[eye_distance: float<br/>Binocular mode]
    
    D --> D1[spawnX, Y, Z: float]
    D --> D2[spawnYaw, Pitch: float]
    D --> D3[gameMode: string]
    
    E --> E1[lidarConfig: LidarConfig]
    E --> E2[soundConfig: SoundConfig]
    E --> E3[requiresDepth: bool]
    E --> E4[blockCollisionKeys: list]
    E --> E5[entityCollisionKeys: list]
    
    F --> F1[structures: repeated Structure]
    F --> F2[items: repeated Item]
    F --> F3[entities: repeated Entity]
    
    style A fill:#e1f5ff
    style B fill:#c8e6c9
    style C fill:#fff9c4
    style D fill:#ffccbc
    style E fill:#e1bee7
    style F fill:#b2dfdb
```

## 컴포넌트 상세 구조

```mermaid
graph TB
    subgraph "Python Package Structure"
        A[craftground/] --> B[__init__.py<br/>make function]
        A --> C[environment/]
        A --> D[environments/]
        A --> E[proto/]
        A --> F[wrappers/]
        A --> G[nbt/]
        A --> H[MinecraftEnv/]
        
        C --> C1[environment.py<br/>CraftGroundEnvironment]
        C --> C2[action_space.py<br/>Action definitions]
        C --> C3[observation_space.py<br/>Observation definitions]
        C --> C4[socket_ipc.py<br/>Socket IPC]
        C --> C5[boost_ipc.py<br/>Shared memory IPC]
        C --> C6[observation_converter.py<br/>Image conversion]
        C --> C7[ipc_interface.py<br/>IPC interface]
        
        D --> D1[base_environment.py]
        D --> D2[find_village_environment.py]
        D --> D3[husk_environment.py]
        D --> D4[... other environments]
        
        E --> E1[action_space_pb2.py]
        E --> E2[observation_space_pb2.py]
        E --> E3[initial_environment_pb2.py]
        
        F --> F1[sb3_environment.py<br/>Stable-Baselines3 wrapper]
        F --> F2[fast_reset.py]
        F --> F3[bimodal.py]
        F --> F4[sound.py]
        F --> F5[vision.py]
    end
    
    subgraph "C++ Native Code"
        I[src/cpp/] --> I1[ipc.cpp<br/>Base IPC functions]
        I --> I2[ipc_boost.cpp<br/>Boost shared memory]
        I --> I3[ipc_cuda.cpp<br/>CUDA support]
        I --> I4[ipc_apple.mm<br/>Apple Metal support]
        I --> I5[ipc.h<br/>Header file]
    end
    
    subgraph "Minecraft Mod Structure"
        H --> H1[src/main/java/]
        H1 --> H2[MinecraftEnv.kt<br/>Main mod class]
        H1 --> H3[MessageIO.kt<br/>IPC interface]
        H1 --> H4[EnvironmentInitializer.kt]
        H1 --> H5[FramebufferCapturer.kt]
        H1 --> H6[ObservationCollector.kt]
        H1 --> H7[ActionExecutor.kt]
        H1 --> H8[TickSynchronizer.kt]
        H1 --> H9[proto/<br/>Protobuf definitions]
    end
    
    style A fill:#e1f5ff
    style I fill:#fff9c4
    style H fill:#c8e6c9
```

## 빌드 및 배포 플로우

```mermaid
graph TB
    A[Source Code] --> B[Python Package]
    A --> C[C++ Extension]
    A --> D[Minecraft Mod]
    
    B --> B1[pyproject.toml<br/>scikit-build-core]
    B --> B2[CMakeLists.txt]
    B --> B3[Build Python wheel]
    
    C --> C1[CMake Configuration]
    C --> C2[pybind11 Binding]
    C --> C3[Compile C++ Code]
    C --> C4[Link Libraries<br/>Boost/CUDA/Apple]
    
    D --> D1[build.gradle]
    D --> D2[Gradle Build]
    D --> D3[Fabric Mod JAR]
    D --> D4[Include in Package]
    
    B3 --> E[Wheel Distribution]
    C4 --> E
    D4 --> E
    
    E --> F[PyPI Package]
    E --> G[GitHub Releases]
    
    style A fill:#e1f5ff
    style E fill:#c8e6c9
    style F fill:#fff9c4
```

## 데이터 변환 파이프라인

```mermaid
graph LR
    A[Minecraft Framebuffer] --> B[FramebufferCapturer]
    B --> C[RGB Bytes]
    
    C --> D{Encoding Mode}
    
    D -->|PNG| E[PNG Encode]
    D -->|RAW| F[Raw Bytes]
    D -->|ZeroCopy| G[Memory Handle]
    D -->|JAX| H[JAX Array]
    
    E --> I[Protobuf Message]
    F --> I
    G --> I
    H --> I
    
    I --> J[IPC Transport]
    
    J --> K{Python Side}
    
    K -->|PNG| L[PNG Decode]
    K -->|RAW| M[Reshape Array]
    K -->|ZeroCopy| N[Create Tensor]
    K -->|JAX| O[JAX Array]
    
    L --> P[numpy.ndarray]
    M --> P
    N --> Q[torch.Tensor<br/>or jax.numpy.ndarray]
    O --> Q
    
    P --> R[RL Agent]
    Q --> R
    
    style A fill:#e1f5ff
    style I fill:#b3e5fc
    style R fill:#c8e6c9
```

이 플로우차트는 CraftGround의 전체 아키텍처와 데이터 흐름을 상세히 보여줍니다. 각 컴포넌트의 역할과 상호작용을 이해하는 데 도움이 됩니다.
