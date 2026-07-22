결론

CraftGround에는 다음 구성이 가장 적합합니다.

1. Minecraft 버전별 Fabric 프로젝트를 독립 Gradle root로 분리
2. Python API와 순수 C++ IPC는 공통 유지
3. 게임 연동 Java/Mixin/렌더 캡처 계층만 버전별 구현
4. PyPI 배포는 공통 패키지 + 버전별 런타임 패키지로 분리
5. CI에서는 Minecraft 버전 × OS × 아키텍처 matrix로 병렬 빌드

즉, 일반적인 Gradle multi-project보다 아래 구조가 안전합니다.

CraftGround/
├── python/                         # 공통 Python API
├── native/ipc/                     # 공통 C++ IPC + pybind11
├── protocol/                       # protobuf/schema/호환성 정의
├── minecraft/
│   ├── mc121/                      # 독립 Gradle root
│   │   ├── gradlew
│   │   ├── gradle/wrapper/
│   │   ├── settings.gradle
│   │   └── build.gradle
│   └── mc262/                      # 독립 Gradle root
│       ├── gradlew
│       ├── gradle/wrapper/
│       ├── settings.gradle
│       └── build.gradle
├── shared-java/                    # Minecraft 클래스를 참조하지 않는 코드만
└── scripts/
    └── build-runtime.py            # 각 Gradle root와 CMake를 조율

⸻

1. Loom의 multi-version 지원 범위

loom { minecraftVersion(...) } DSL은 사용하지 않는다

Loom에서 Minecraft 버전은 일반적으로 loom 블록이 아니라 Gradle dependency로 지정합니다.

dependencies {
    minecraft "com.mojang:minecraft:${project.minecraft_version}"
    mappings "net.fabricmc:yarn:${project.yarn_mappings}:v2"
}

공식 Loom 옵션도 Minecraft 버전을 minecraft dependency configuration으로 지정한다고 설명합니다. 별도의 공식 loom.minecraftVersion() variant DSL은 없습니다.  

따라서 다음처럼 property 하나를 바꾸어 한 번에 하나의 버전을 선택하는 것은 가능합니다.

./gradlew build -Pminecraft_version=1.21

하지만 이것은 Android product flavor 같은 multi-variant 시스템이 아닙니다. 한 빌드 invocation에서 서로 다른 Minecraft classpath와 mappings를 가진 두 JAR을 자연스럽게 만드는 표준 기능은 아닙니다.

versionRange()도 빌드용 Loom DSL이 아니다

Fabric에는 크게 두 종류의 버전 지정이 있습니다.

* 빌드 시점의 정확한 Minecraft dependency
* fabric.mod.json에 기록하는 런타임 호환 범위

예를 들어:

{
  "depends": {
    "minecraft": ">=1.21 <1.22"
  }
}

이 범위는 Loader가 모드 호환성을 검사하기 위한 metadata입니다. 서로 다른 Minecraft API에 대해 하나의 source set을 여러 번 컴파일해 주는 기능이 아닙니다.

CraftGround처럼 Mixin target과 렌더링 내부 구조가 달라지는 경우, 넓은 version range를 선언해서 해결할 수 없습니다.

Gradle variants는 이론적으로 가능하지만 권장되지 않는다

직접 다음을 구현할 수는 있습니다.

* mc121CompileClasspath
* mc262CompileClasspath
* 버전별 source set
* 버전별 RemapJarTask
* 버전별 mappings
* 버전별 resources

하지만 Loom 자체가 한 프로젝트에 하나의 Minecraft 개발 환경을 구성하는 것을 기본 전제로 합니다. Loom은 configured Minecraft version을 받아 해당 JAR, mappings, remap 환경을 준비합니다.  

특히 CraftGround는 다음까지 다릅니다.

* Loom plugin ID와 버전
* Gradle wrapper
* mappings 체계
* Mixin targets
* Fabric API
* renderer integration
* native capture 코드

따라서 custom variant를 만드는 비용이 별도 프로젝트 두 개를 유지하는 비용보다 큽니다.

splitEnvironmentSourceSets()는 multi-version 기능이 아니다

이 기능은 다음을 분리합니다.

common/server-safe code
client-only code

버전별 코드를 분리하는 기능은 아닙니다.

loom {
    splitEnvironmentSourceSets()
    mods {
        craftground {
            sourceSet sourceSets.main
            sourceSet sourceSets.client
        }
    }
}

빌드 결과도 여전히 해당 Minecraft 버전용 단일 JAR입니다. Fabric 문서 역시 이 기능을 client/common compile-time isolation으로 설명합니다.  

Classpath Groups도 하나의 모드를 구성하는 여러 source set이나 multi-project classpath를 Loader에게 알려주기 위한 기능이지, Minecraft 버전 차이를 추상화하는 기능은 아닙니다.

⸻

2. 가장 적절한 Gradle 구조

일반 subproject보다 독립 Gradle root가 낫다

질문의 Pattern A는 겉보기에는 적절하지만 중요한 문제가 있습니다.

Gradle multi-project build 하나는 기본적으로 하나의 Gradle wrapper 버전으로 실행됩니다.

그런데 현재 요구사항은:

mc121: Loom 1.6 + Gradle 8.x
mc262: Loom 1.17 + Gradle 9.5.1

입니다. Fabric은 26.2 개발에 Loom 1.17과 Gradle 9.5.1 사용을 안내하고 있습니다.  

따라서 다음 구조는 위험합니다.

MinecraftEnv/
├── gradlew                  # 어느 버전이어야 하는가?
├── mc121/
└── mc262/

Gradle 9.5.1에서 Loom 1.6이 정상 동작한다는 보장이 없고, Gradle 8 wrapper에서는 Loom 1.17/26.2 설정이 맞지 않을 수 있습니다.

권장 구조

minecraft/
├── mc121/
│   ├── gradlew
│   ├── gradle/wrapper/gradle-wrapper.properties
│   ├── settings.gradle
│   ├── build.gradle
│   └── src/
└── mc262/
    ├── gradlew
    ├── gradle/wrapper/gradle-wrapper.properties
    ├── settings.gradle
    ├── build.gradle
    └── src/

그리고 상위 빌드 스크립트가 두 wrapper를 각각 호출합니다.

./minecraft/mc121/gradlew -p minecraft/mc121 build
./minecraft/mc262/gradlew -p minecraft/mc262 build

이는 별도 repository 두 개가 아니라 monorepo 안의 독립 Gradle builds입니다.

Gradle composite build로 묶는 방법도 있지만, CraftGround 패키징에서는 단순한 orchestration script가 더 예측 가능합니다.

⸻

3. Pattern A/B/C 비교

패턴	평가	장점	문제
Subproject per version	조건부 가능	한 Gradle task graph, 공통 convention 적용	서로 다른 Gradle/Loom 세대 사용 곤란
Single project variants	비추천	표면상 파일 수가 적음	Loom 환경, mappings, Mixin AP, remap task를 직접 복제해야 함
독립 root projects	권장	wrapper와 Loom 완전 격리, 문제 재현 쉬움	orchestration script 필요
완전 별도 repository	초기에는 비추천	릴리스 주기 완전 독립	코드 동기화와 atomic protocol 변경이 어려움

CraftGround에서는 Pattern C의 monorepo 변형이 가장 좋습니다.

하나의 Git repository
+
버전별 독립 Gradle root
+
공통 Python/C++/protocol

공통 Java 코드는 제한적으로 공유

공유 가능한 코드:

* protobuf message handling
* byte-buffer utility
* action/observation validation
* IPC framing
* logging utility
* 버전 독립 configuration model

공유하면 안 되는 코드:

* Minecraft/Yarn/Mojang class import
* Mixin target
* renderer hook
* framebuffer/texture 접근
* GUI/HUD 접근
* registry integration

예를 들어:

shared-java/
└── src/main/java/
    └── IPCProtocol.java
minecraft/mc121/
└── Minecraft121FrameCapture.java
minecraft/mc262/
└── Minecraft262FrameCapture.java

shared-java는 plain Java library로 빌드하거나 source directory로 포함할 수 있습니다. 장기적으로는 plain Maven artifact가 가장 깨끗합니다.

⸻

4. Python 패키지 배포 권고

추천: Option C를 변형한 core/runtime 분리

다음과 같이 구성하는 것이 가장 균형이 좋습니다.

pip install craftground
pip install craftground-runtime-mc121
pip install craftground-runtime-mc262

또는 extras UX를 추가합니다.

pip install "craftground[mc121]"
pip install "craftground[mc262]"
pip install "craftground[all]"

사용 코드는 계속 동일하게 유지합니다.

import craftground
env = craftground.make(mc_version="26.2")

중요한 점은 버전별 배포 패키지가 craftground Python namespace 전체를 각각 구현해서는 안 된다는 것입니다. 그것은 파일 충돌과 uninstall 문제를 일으킬 수 있습니다.

대신 runtime package는 asset package로 만듭니다.

craftground-runtime-mc262/
└── craftground_runtime_mc262/
    ├── manifest.json
    ├── craftground-mc262.jar
    └── native/

공통 craftground는 Python package metadata나 entry point를 통해 설치된 runtime을 찾습니다.

옵션 비교

옵션	빌드 시간	배포 크기	런타임 복잡도	유지보수	평가
A. 모든 버전 내장	매우 큼	매우 큼	중간	중간~높음	버전 2개일 때는 가능
B. 같은 PyPI 이름의 release 버전으로 구분	낮음	낮음	낮음	중간	Python 패키지 버전과 MC 버전이 뒤섞임
C. 버전별 package 이름	중간	낮음	낮음~중간	낮음	권장, core/runtime 분리 필요
D. 외부 mod path	가장 낮음	가장 작음	높음	낮음	개발자용 fallback으로 적합

Option B의 문제

craftground==2.6.26.2

처럼 Python distribution version에 Minecraft version을 직접 넣으면 다음 의미가 섞입니다.

* CraftGround 자체 버전
* protocol 버전
* Minecraft target 버전
* packaging bugfix 버전

예를 들어 “CraftGround 3.1.0의 MC 1.21 runtime 두 번째 패치”를 표현하기 어려워집니다.

더 나은 표기는:

craftground                  3.1.0
craftground-runtime-mc121    3.1.0.2
craftground-runtime-mc262    3.1.0.1

또는 runtime metadata에 별도로 기록하는 방식입니다.

연구 환경 관점의 최종 선택

기본 배포는:

pip install "craftground[mc262]"

고급 사용자는:

craftground.make(
    mc_version="26.2",
    mod_path="/custom/craftground.jar",
)

를 사용할 수 있게 하는 것이 좋습니다.

즉, Option C를 기본으로 하고 Option D를 override로 제공하는 구성이 적합합니다.

⸻

5. C++ IPC와 pybind11의 버전 독립성

src/cpp IPC는 조건부로 공통 binary 사용 가능

다음 조건을 모두 만족한다면 같은 binary를 사용할 수 있습니다.

* Minecraft class나 JNI class layout을 직접 참조하지 않음
* OpenGL/Vulkan symbol을 링크하지 않음
* protobuf/schema가 동일함
* shared-memory layout이 동일함
* message framing과 enum 값이 동일함
* Java side JNI method signature가 동일함

그렇다면 Minecraft 1.21과 26.2의 차이는 IPC binary에 보이지 않습니다.

Minecraft mod
    ↓ protocol bytes/shared memory
common C++ IPC
    ↓
common pybind11 extension
    ↓
Python

따라서 질문한 hybrid 아키텍처는 가능하며, 실제로 권장됩니다.

단, 두 C++ 계층을 구분해야 한다

레포지토리 설명상 C++이 두 위치에 있습니다.

src/cpp/                         # Python ↔ JVM IPC
MinecraftEnv/src/main/cpp/       # JNI frame capture/OpenGL

첫 번째는 버전 독립적으로 유지할 가능성이 큽니다.

두 번째가 glReadPixels, framebuffer ID, OpenGL context에 의존한다면 버전 독립적이지 않습니다. 특히 Vulkan backend에서는 OpenGL library로 같은 동작을 수행할 수 없습니다.

즉:

공통 가능:
- IPC
- pybind11
- protobuf
- shared-memory transport
분리 필요:
- graphics capture JNI
- backend synchronization
- GPU readback
- renderer hook

protocol handshake를 추가해야 한다

버전별 runtime을 도입하기 전에 다음 정보를 시작 시 교환하는 것이 좋습니다.

{
  "protocol_version": 3,
  "minecraft_version": "26.2",
  "craftground_mod_version": "3.1.0",
  "render_backend": "vulkan",
  "capabilities": [
    "rgb_frame",
    "depth_frame",
    "sound_observation"
  ]
}

Python은 protocol incompatibility를 즉시 거부해야 합니다.

⸻

6. Vulkan 프레임 캡처 범위

Java/Kotlin 전체 재작성은 아니다

다음은 그대로 유지할 가능성이 높습니다.

* action injection
* keyboard/mouse emulation
* world state observation
* entity query
* sound observation
* protobuf serialization
* IPC transport
* lifecycle 관리

크게 변경되는 영역은 renderer와 직접 연결된 부분입니다.

Mixin injection point
frame completion timing
render target 접근
GPU → CPU readback
buffer synchronization
pixel format/row stride
native library loading

따라서 “모드 전체 재작성”보다는 프레임 캡처 subsystem 재작성에 가깝습니다.

다만 CraftGround에서 시각 관찰이 핵심이므로, 변경 코드의 줄 수보다 시스템 위험도는 높습니다.

gl*을 단순히 vk*로 치환하면 안 된다

OpenGL에서는 보통:

현재 framebuffer bind
glReadPixels
CPU 메모리로 동기 readback

이 가능합니다.

Vulkan에서는 일반적으로:

이미지 layout transition
GPU image → staging buffer copy
pipeline barrier
fence/semaphore 동기화
mapped host-visible memory 접근

이 필요합니다.

따라서 이것은 API 이름 변경이 아니라 resource lifecycle과 synchronization 모델의 변경입니다.

가능한 추상화 경계

CraftGround가 직접 com.mojang.blaze3d.opengl과 vulkan 구현을 동일 인터페이스로 감싸는 것보다, 가능한 한 Minecraft의 backend-neutral Blaze3D abstraction 위에서 캡처하는 것이 좋습니다.

interface FrameCaptureBackend {
    CapturedFrame capture(RenderCaptureContext context);
}

구현은 예를 들어:

Mc121OpenGlFrameCapture
Mc262Blaze3dFrameCapture
Mc262OpenGlFallbackCapture
Mc262VulkanFrameCapture

하지만 OpenGLFramebuffer와 VulkanImage를 그대로 공통 interface에 노출해서는 안 됩니다. 공통 interface는 결과 중심이어야 합니다.

record CapturedFrame(
    ByteBuffer pixels,
    int width,
    int height,
    PixelFormat format,
    long frameNumber
) {}

26.2에서는 우선 OpenGL backend로 포팅하는 전략도 가능

26.2는 기본 OpenGL과 실험적 Vulkan을 선택할 수 있으며, Fabric은 raw OpenGL 사용 모드는 마이그레이션이 필요하다고 안내합니다.  

단계적으로 다음이 현실적입니다.

1. 26.2 게임 코드/Mixin 변경을 먼저 해결
2. 26.2 OpenGL backend에서 Blaze3D-compatible capture 구현
3. backend-neutral capture contract 확정
4. Vulkan readback 구현
5. 두 backend의 픽셀 동일성 및 latency 검증
6. OpenGL 제거 전에 Vulkan을 기본으로 전환

Fabric 문서도 raw OpenGL 대신 Minecraft 렌더링 시스템을 사용해야 하며, 1.21.6 이후 RenderPipeline/RenderState 중심의 대규모 변경이 진행됐다고 설명합니다.  

⸻

7. Option A를 선택할 때 wheel 구조

native IPC가 공통이면 버전별 복제하지 않는다

IPC binary가 정말 Minecraft 버전 독립적이라면 다음처럼 구성해야 합니다.

craftground/
├── _native/
│   ├── craftground_ipc.so
│   └── platform.json
├── runtimes/
│   ├── mc121/
│   │   ├── manifest.json
│   │   ├── craftground-mc121.jar
│   │   └── capture/
│   │       └── libcraftground_capture_gl.so
│   └── mc262/
│       ├── manifest.json
│       ├── craftground-mc262.jar
│       └── capture/
│           ├── libcraftground_capture_gl.so
│           └── libcraftground_capture_vk.so
└── ...

공통 IPC를 다음처럼 두 번 복제할 이유는 없습니다.

mc121/native-lib.so
mc262/native-lib.so

단, 해당 native-lib가 renderer/JNI capture도 포함한 monolithic library라면 버전별로 분리해야 합니다. 가장 좋은 해결은 native library 자체를 둘로 나누는 것입니다.

libcraftground_ipc
libcraftground_capture_gl
libcraftground_capture_vk

py3-none-any.whl은 사용할 수 없다

pybind11 extension이나 .so/.dylib/.dll을 포함하면 wheel은 platform-specific입니다.

예:

craftground-3.1.0-cp312-cp312-manylinux_2_28_x86_64.whl
craftground-3.1.0-cp312-cp312-macosx_14_0_arm64.whl
craftground-3.1.0-cp312-cp312-win_amd64.whl

Stable ABI를 사용한다면 Python version tag 범위를 줄일 수는 있지만 none-any는 아닙니다.

동일 Python build에서 두 Gradle build 실행 가능

가능합니다. 다만 “하나의 Gradle build”가 아니라 packaging orchestrator가 두 독립 Gradle build를 순차 호출하는 방식이어야 합니다.

subprocess.run([
    "minecraft/mc121/gradlew",
    "-p", "minecraft/mc121",
    "buildRuntime",
], check=True)
subprocess.run([
    "minecraft/mc262/gradlew",
    "-p", "minecraft/mc262",
    "buildRuntime",
], check=True)
subprocess.run([
    "cmake",
    "--build", "build/native",
], check=True)

그러나 pip install 시점마다 Minecraft, mappings, Fabric API와 Gradle dependency를 받아 두 mod를 빌드하는 방식은 피하는 것이 좋습니다.

권장 방식은:

CI에서 JAR/native runtime을 미리 빌드
→ wheel에 artifact 포함
→ 사용자는 binary wheel 설치

소스 배포용 sdist는 fallback으로만 유지하는 편이 좋습니다.

⸻

8. CI/CD 전략

Matrix를 두 계층으로 나눈다

Mod build matrix

Java JAR이 OS 독립적이고 renderer native library가 없다면:

strategy:
  matrix:
    mc:
      - id: mc121
        path: minecraft/mc121
      - id: mc262
        path: minecraft/mc262

각 job은 해당 프로젝트의 wrapper를 사용합니다.

- run: ./${{ matrix.mc.path }}/gradlew
       -p ${{ matrix.mc.path }}
       build

Native wheel matrix

strategy:
  matrix:
    os: [ubuntu-latest, macos-14, windows-latest]
    python: ["3.10", "3.11", "3.12", "3.13"]

렌더 capture native library가 버전별이면 여기에 mc dimension을 추가합니다.

OS × architecture × Python × MC runtime

다만 artifact를 단계적으로 재사용하여 조합 폭발을 줄여야 합니다.

권장 pipeline

1. protocol-tests
   └── protobuf/schema compatibility
2. build-mod-mc121
   └── mc121 JAR
3. build-mod-mc262
   └── mc262 JAR
4. build-native-ipc
   └── OS/arch별 공통 native binary
5. build-capture-native
   └── 필요한 경우 MC/backend/OS별 capture binary
6. assemble-runtime-packages
   ├── craftground-runtime-mc121
   └── craftground-runtime-mc262
7. build-core-wheels
   └── craftground
8. integration-tests
   ├── mc121 + OpenGL
   ├── mc262 + OpenGL
   └── mc262 + Vulkan
9. publish

하나의 universal wheel보다 별도 runtime package 권장

하나의 wheel에 모두 넣으면:

* 모든 사용자가 불필요한 JAR과 GPU backend를 다운로드
* 한 버전의 runtime hotfix에도 전체 wheel 재배포
* 플랫폼별 native 조합으로 wheel 크기 증가
* 설치 후 어떤 runtime이 활성화됐는지 불명확
* 향후 MC 버전이 늘어날수록 선형으로 비대해짐

버전별 runtime package를 사용하면 core Python release와 runtime release를 분리할 수 있습니다.

⸻

최종 권고안

craftground                    # Python API + pybind11 IPC
craftground-runtime-mc121      # 1.21 Fabric JAR + OpenGL capture
craftground-runtime-mc262      # 26.2 Fabric JAR + GL/Vulkan capture

사용자 UX:

pip install "craftground[mc262]"
env = craftground.make(
    mc_version="26.2",
    render_backend="vulkan",
)

내부 repository:

minecraft/mc121/    # Loom 1.6, 자체 wrapper
minecraft/mc262/    # Loom 1.17, Gradle 9.5.1, 자체 wrapper
native/ipc/         # 공통
python/             # 공통
protocol/           # 공통 + version handshake

핵심 경계는 다음과 같습니다.

공통:
Python API
Gymnasium environment
protobuf
shared-memory protocol
C++ IPC
pybind11
버전별:
Fabric bootstrap
Mixin targets
Minecraft class access
GUI/registration hooks
render completion hook
backend별:
OpenGL frame readback
Vulkan frame readback
GPU synchronization

따라서 CraftGround의 26.2 마이그레이션은 전체 시스템 재작성은 아니지만, 렌더 캡처 계층은 별도 backend 구현에 가까운 수준으로 재설계해야 합니다. Loom의 multi-version variant 기능으로 해결하려 하기보다, 빌드 환경을 독립시키고 protocol 경계를 안정화하는 것이 장기적으로 훨씬 안전합니다.