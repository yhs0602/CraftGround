1. 가장 추천하는 선택지

빌드 구조

하나의 monorepo 안에 Minecraft 버전별 독립 Gradle project를 두는 방식을 가장 추천합니다.

CraftGround/
├── minecraft/
│   ├── mc121/              # 독립 Gradle root
│   │   ├── settings.gradle
│   │   ├── build.gradle
│   │   ├── gradlew
│   │   └── gradle/wrapper/
│   └── mc262/              # 독립 Gradle root
│       ├── settings.gradle
│       ├── build.gradle
│       ├── gradlew
│       └── gradle/wrapper/
├── shared/
│   ├── protocol-java/
│   └── native-ipc/
└── python/

정확히는:

* 별도 저장소 두 개: 비추천
* 하나의 Gradle project 안에서 variant 구성: 비추천
* 하나의 monorepo 안에서 독립 Gradle build 두 개: 가장 추천

이유는 1.21과 26.2가 사용하는 Gradle·Loom 세대가 다르기 때문입니다. 같은 multi-project build에 넣으면 모든 subproject가 기본적으로 하나의 Gradle wrapper로 실행됩니다. 반면 독립 build는 각 디렉터리가 자신의 wrapper와 Loom 버전을 가질 수 있습니다.

./minecraft/mc121/gradlew -p minecraft/mc121 build
./minecraft/mc262/gradlew -p minecraft/mc262 build

Gradle composite build는 여러 독립 build를 한 작업공간에서 함께 개발하기 위한 공식 기능이며, 각각을 독립적으로 열 수도 있고 하나의 composite로 묶을 수도 있습니다.  

pip 배포

앞서 제시한 것 중에는 Option C의 개선형을 추천합니다.

pip install "craftground[mc121]"
pip install "craftground[mc262]"

내부적으로는:

craftground
craftground-runtime-mc121
craftground-runtime-mc262

로 분리합니다.

Python API는 공통으로 유지합니다.

env = craftground.make(mc_version="26.2")

외부 mod 경로 지정은 개발 및 디버깅용 override로 함께 지원하면 됩니다.

⸻

2. 별도 Gradle project일 때 Language Server 기능

결론

자동완성, 정의로 이동, 레퍼런스 찾기 모두 정상적으로 사용할 수 있습니다.

단, IDE가 두 Gradle build를 모두 import하도록 구성해야 합니다.

IntelliJ IDEA에서는 composite build를 import하면 각 included build의 subproject가 IDE module로 들어오므로, 여러 독립 Gradle build의 소스를 한 workspace에서 함께 탐색할 수 있습니다.  

권장 방식은 최상위에 개발 편의용 composite build를 두는 것입니다.

CraftGround/
├── settings.gradle.kts       # IDE/composite 전용
├── minecraft/
│   ├── mc121/
│   └── mc262/
└── shared/
    └── protocol-java/

최상위 settings.gradle.kts:

rootProject.name = "craftground-workspace"
includeBuild("minecraft/mc121")
includeBuild("minecraft/mc262")
includeBuild("shared/protocol-java")

이 최상위 프로젝트를 IntelliJ에서 열면 IDE가 대략 다음 module들을 인식합니다.

craftground-mc121
craftground-mc262
craftground-protocol

그 결과:

* 각 버전의 Minecraft/Yarn 클래스 자동완성
* Mixin target 클래스 탐색
* 정의로 이동
* 사용처 찾기
* 공통 모듈에서 각 구현으로 이동
* 버전별 compile error 표시

가 가능합니다.

Gradle도 composite build를 “독립 build를 나누어 유지하면서 IDE에서 함께 개발”하는 용도로 공식 설명합니다.  

⸻

공통 로직 공유 방법

공통 로직은 두 종류로 구분해야 합니다.

A. 실제 Java 공통 코드

별도의 plain Java library project로 두는 것이 좋습니다.

shared/protocol-java/
├── settings.gradle.kts
├── build.gradle.kts
└── src/main/java/
    └── io/craftground/shared/
        ├── IPCProtocol.java
        ├── FrameMetadata.java
        └── ObservationCodec.java

각 Minecraft 프로젝트는 동일한 Maven coordinate로 의존합니다.

dependencies {
    implementation("io.craftground:craftground-protocol:1.0.0")
}

개발 중에는 composite build의 dependency substitution이 로컬 소스를 자동 연결할 수 있습니다.

// 최상위 settings.gradle.kts
includeBuild("shared/protocol-java") {
    dependencySubstitution {
        substitute(module("io.craftground:craftground-protocol"))
            .using(project(":"))
    }
}

그러면 공통 코드 수정 시:

* 별도 publish 불필요
* 두 MC 프로젝트에서 즉시 반영
* 공통 타입으로 정의 이동 가능
* 공통 메서드의 사용처를 두 모듈에서 찾을 수 있음

Gradle composite build는 외부 dependency를 로컬 project source로 치환하여 라이브러리와 consumer를 동시에 개발하는 방식을 공식 지원합니다.  

B. 공통 Gradle 설정

소스 코드와 build 설정은 분리해야 합니다.

공통 Gradle 설정은 convention plugin으로 관리하는 것이 좋습니다.

build-logic/
├── settings.gradle.kts
└── src/main/kotlin/
    ├── craftground.java-conventions.gradle.kts
    └── craftground.native-conventions.gradle.kts

각 버전 프로젝트에서:

pluginManagement {
    includeBuild("../../build-logic")
}
plugins {
    id("craftground.java-conventions")
}

Gradle은 build 간 공통 설정을 공유할 때 composite build-logic 및 convention plugin 사용을 권장합니다.  

다만 Loom 자체 설정은 버전별로 남기는 편이 좋습니다.

공통화하기 좋은 것:
- Java toolchain 21
- compiler flags
- 테스트 설정
- protobuf 생성
- native artifact 복사 규칙
- formatting/checkstyle
버전별 유지할 것:
- Loom plugin version
- Minecraft dependency
- Yarn mappings
- Fabric Loader/API
- Mixin configuration
- remapJar 설정

⸻

레퍼런스 찾기의 주의점

같은 이름의 버전별 클래스

다음처럼 두 프로젝트에 동일한 FQCN을 만들 수도 있습니다.

mc121:
io.craftground.capture.MinecraftFrameCapture
mc262:
io.craftground.capture.MinecraftFrameCapture

각각은 별도 module classpath라서 컴파일상 문제는 없습니다. 다만 IDE 전체 검색에서는 두 정의가 함께 표시될 수 있습니다.

더 명확하게 하려면 구현 이름을 구분하는 편이 좋습니다.

Mc121FrameCapture
Mc262FrameCapture

또는 공통 인터페이스와 버전별 구현으로 나눕니다.

// shared
public interface FrameCapture {
    CapturedFrame capture();
}
// mc121
public final class OpenGl121FrameCapture implements FrameCapture {
}
// mc262
public final class Blaze3d262FrameCapture implements FrameCapture {
}

이 구조에서는 공통 인터페이스의 “Find Usages”를 실행하면 두 버전 구현과 호출부를 함께 추적하기 쉽습니다.

Yarn mapping이 서로 다를 때

각 Gradle module은 자신의 Minecraft dependency와 mappings를 별도로 갖습니다.

따라서 IDE는:

mc121 module → Minecraft 1.21 + 해당 Yarn mappings
mc262 module → Minecraft 26.2 + 해당 mappings

를 각각 다른 classpath로 인덱싱합니다.

1.21 코드에서 26.2 메서드가 자동완성되는 식으로 섞이지 않습니다. 오히려 단일 variant 프로젝트보다 버전별 API 차이가 명확하게 드러납니다.

VS Code 사용 시

VS Code Java language server도 여러 Gradle project를 workspace에 넣으면 동작하지만, Loom/Mixin 개발 경험은 보통 IntelliJ가 더 안정적입니다.

멀티 루트 workspace를 사용합니다.

{
  "folders": [
    { "path": "minecraft/mc121" },
    { "path": "minecraft/mc262" },
    { "path": "shared/protocol-java" }
  ]
}

다만 대형 Minecraft dependency를 두 벌 인덱싱하므로 메모리와 초기 인덱싱 시간이 증가합니다.

⸻

최종 권장 구조

CraftGround/
├── settings.gradle.kts              # IDE용 composite workspace
├── build-logic/                     # 공통 Gradle convention
│
├── shared/
│   ├── protocol-java/               # 순수 Java 공통 코드
│   └── native-ipc/                  # 공통 C++ IPC
│
├── minecraft/
│   ├── mc121/                       # 독립 Gradle wrapper
│   │   └── src/main/java/
│   └── mc262/                       # 독립 Gradle wrapper
│       └── src/main/java/
│
├── python/                          # 공통 Python API
└── scripts/
    └── build-all.py

이 구조의 결과는 다음과 같습니다.

항목	결과
Gradle/Loom 버전 격리	완전 격리
공통 코드 수정	두 프로젝트에 즉시 반영
자동완성	각 MC 버전 classpath 기준 정상 작동
정의로 이동	공통 및 버전별 코드 모두 가능
레퍼런스 찾기	composite workspace 전체에서 가능
개별 모듈 디버깅	각 Gradle root를 별도로 실행
전체 CI 빌드	상위 script 또는 matrix로 실행
향후 MC 버전 추가	minecraft/mcXXX 추가

따라서 별도 Gradle 프로젝트 때문에 IDE 기능이 희생되지는 않습니다. 오히려 독립 classpath 덕분에 각 버전에서 유효한 Minecraft API가 명확히 분리되어 잘못된 자동완성을 방지할 수 있습니다.