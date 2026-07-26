# 26.2 마이그레이션 — Phase 2 실행 계획 (mc262 실구현)

Phase 1(빌드 구조 재정비, `26_2_work_plan.md`)이 끝나고 mixin 9개가 포팅된 상태에서,
남은 14개 mixin과 관측/캡처 파이프라인을 어떻게 구현할지에 대한 확정 계획.

근거가 되는 26.2 디컴파일 소스 분석은 `26_2_mixin_deep_dive.md` 참고(로컬 전용, gitignore).

---

## 0. 설계 원칙

RL 환경으로서 지켜야 할 semantics는 mc121과 동일하게 유지한다:

```
Action 수신
 → Client input injection
 → Integrated Server 1 tick
 → Tick synchronization
 → Client state updated
 → (기존) Minecraft render pipeline
 → Capture
 → Observation 반환
```

핵심 원칙: **"렌더 파이프라인을 다시 실행"하지 않는다. "정확한 tick 이후 정확한 frame을
잡는다."**

관측 생성 로직이 렌더 파이프라인을 직접 소유하게 되면 (a) Minecraft frame lifecycle과
어긋나고 (b) mc121/26.2 semantics 차이가 커지며 (c) 불필요한 추가 렌더 비용이 발생한다.
따라서 26.2의 `extract → render` 분리에 대응하는 방법은 **강제 재렌더가 아니라 캡처
지점을 올바른 frame boundary로 옮기는 것**이다.

---

## 1. 확정된 설계 결정

| ID | 결정 | 상태 |
|----|------|------|
| D1 | observation마다 `extract`/`render` 강제 재실행 | ❌ 채택 안 함 |
| D1' | 동기화 강화 + 캡처를 post-render frame boundary로 이동 | ✅ 채택 |
| D1'' | 동기화 지점에서 `PacketProcessor` 명시적 드레인 | ✅ 채택 (신규) |
| D2 | GL 백엔드가 아니면 시작 시 fail-fast | ✅ 채택 |
| D3 | FBO 기반 API 제거, texture 기반 abstraction (`captureTexture(textureId)`) | ✅ 채택 |
| D3' | FBO는 native가 직접 소유/캐시 (zerocopy 경로 보존용) | ✅ 채택 (수정) |
| D3'' | `glGetTextureImage`는 CPU path 한정, 동작 확보 후 최적화 단계로 | ✅ 채택 |
| D4 | custom entity(`RealisticHuman`, OBJ) 이번 범위 제외 | ✅ 채택 |
| D5 | 서버 tick 제한 해제는 바닐라 `TickRateManager`로 | ✅ 채택 |
| D5' | **클라이언트 tick 고정(`TickSpeedMixin`)은 유지** | ✅ 채택 (수정) |

### 1.1 D1'' — 왜 PacketProcessor 드레인이 필요한가

26.2에서 클라이언트 패킷 처리는 concurrent queue를 통해 지연 실행된다:

```java
// net/minecraft/network/PacketProcessor.java
private final Queue<ListenerAndPacket<?>> packetsToBeHandled = Queues.newConcurrentLinkedQueue();

public void processQueuedPackets() {
    if (!this.closed) {
        while (!this.packetsToBeHandled.isEmpty()) {
            this.packetsToBeHandled.poll().handle();
        }
    }
}
```

그리고 이 드레인은 `Minecraft.runTick`의 **맨 앞**(`Minecraft.java:1173`), 즉 `tick()`
루프보다 **위**에서만 호출된다.

결과적으로 통합 서버가 tick N에서 보낸 패킷은 큐에 쌓인 채로 있다가 **다음 `runTick`의
시작**에서야 `ClientLevel`에 반영된다. 따라서 `END_WORLD_TICK`에서 서버 tick N 완료를
기다린 뒤 그대로 render/capture하면, 캡처된 프레임은 서버 tick N의 권위 상태를 반영하지
않는다(클라 예측 상태만 반영). 이는 재렌더 여부와 무관한 별개의 결함이다.

**해결**: 동기화 지점에서 큐를 명시적으로 드레인한다. `Minecraft.packetProcessor()`는
public getter(`Minecraft.java:3016`), `processQueuedPackets()`도 public이므로 mixin 불필요.

```kotlin
tickSynchronizer.waitForServerTickCompletion()
client.packetProcessor().processQueuedPackets()   // ← 추가
```

**단, 이 드레인만으로는 부족하다.** 1.3절 참고.

### 1.3 패킷 도착 보장 문제 (소스 확인 완료)

`ServerTickEvents.END_SERVER_TICK` 시점에 해당 tick의 패킷이 이미 클라이언트 큐에 들어와
있는지 확인한 결과 — **보장되지 않는다. 레이스다.**

**(1) 통합 서버의 로컬 연결도 별도 스레드풀을 쓴다**

```java
// net/minecraft/server/network/EventLoopGroupHolder.java
private static final EventLoopGroupHolder LOCAL = new EventLoopGroupHolder("Local", LocalChannel.class, LocalServerChannel.class) {
    @Override protected IoHandlerFactory ioHandlerFactory() { return LocalIoHandler.newFactory(); }
};

private EventLoopGroup createEventLoopGroup() {
    return new MultiThreadIoEventLoopGroup(this.createThreadFactory(), this.ioHandlerFactory());
}
```

서버 스레드도 클라 게임 스레드도 아닌 제3의 netty 스레드풀이다.

**(2) 서버의 `send()`는 쓰기를 예약만 한다 (비동기 hop #1)**

```java
// net/minecraft/network/Connection.java:298
private void sendPacket(final Packet<?> packet, final @Nullable ChannelFutureListener listener, final boolean flush) {
    ...
    if (this.channel.eventLoop().inEventLoop()) {
        this.doSendPacket(packet, listener, flush);
    } else {
        this.channel.eventLoop().execute(() -> this.doSendPacket(packet, listener, flush));  // ← 서버 스레드는 이 분기
    }
}
```

**(3) 클라 수신도 큐에 적재만 한다 (비동기 hop #2)**

`ClientPacketListener`의 **119개 핸들러 전부**가 아래 패턴을 거친다:

```java
PacketUtils.ensureRunningOnSameThread(packet, this, this.minecraft.packetProcessor());
```

```java
// net/minecraft/network/protocol/PacketUtils.java
if (!packetProcessor.isSameThread()) {
    packetProcessor.scheduleIfPossible(listener, packet);
    throw RunningOnDifferentThreadException.RUNNING_ON_DIFFERENT_THREAD;
}
```

**종합 경로:**

```
Server thread ──send()──▶ [netty local event loop] ──write──▶ [peer pipeline]
                 hop #1                                hop #2
  ──scheduleIfPossible──▶ ConcurrentLinkedQueue ──▶ (game thread) processQueuedPackets()
```

`END_SERVER_TICK`은 서버 스레드에서 발생하므로 그 시점에 hop #1조차 실행 전일 수 있다.

**성격**: 26.2 회귀가 아니라 mc121에도 있던 기존 레이스다. 다만 mc121은 관측이 어차피 한
프레임 뒤처져 있어 가려져 있었고, W1으로 frame boundary를 고치면 이것이 남은 유일한
staleness 원인이 된다. 타이밍 의존이라 **재현 안 되는 간헐적 오염**으로 나타난다.
실무적으로는 `waitForServerTickCompletion()` 이후 캡처 + protobuf + IPC + 파이썬 액션 대기
동안 수백 µs~ms가 흘러 대개 도착하지만, 부하/스케줄링에 따라 샌다.

**기존 `TickSynchronizer` 락은 이 문제를 덮지 못한다.** 락의 참여자는 클라이언트 게임
스레드와 서버 스레드 **둘뿐**이고, 패킷이 경유하는 netty local event loop 스레드풀은
비참여자다. 락이 주는 것은 *메모리 가시성*(서버 스레드가 `notifyClientSendObservation()`
이전에 쓴 메모리는 클라가 `waitForServerTickCompletion()` 이후 확실히 봄)이지,
*제3 스레드의 작업 실행*이 아니다. `channel.eventLoop().execute(...)`는 netty 큐에 작업을
등록만 하며 락은 netty 스레드를 재촉하지 못한다.

**관측 소스에 따라 필요 여부가 갈린다 (중요):**

| 관측 소스 | 락만으로 충분한가 |
|---|---|
| 서버 상태 직접 읽기 (`ServerPlayer` 등) | ✅ happens-before로 **완전 보장**. 패킷 무관 |
| 클라 상태 읽기 (`ClientLevel` / `LocalPlayer`) | ❌ 패킷 배달 의존 |
| 렌더링된 이미지 | ❌ `ClientLevel` → extract → render 경유라 **회피 불가** |

현재 `sendObservation`은 `player.pos` / `player.health` /
`player.hungerManager.foodLevel` 등을 전부 **클라 플레이어**에서 읽는다. 코드에 이미
`// TODO: Use server player stats directly instead of client player stats` 주석이 있는데,
**이 TODO가 사실상 본 문제의 해법**이다. 수치 관측을 서버 엔티티에서 읽도록 옮기면 락만으로
완전 보장되고 배리어가 불필요해진다.

따라서 **배리어(W1-b)가 진짜 필요한 대상은 렌더링된 이미지 하나**다.

**대응 방안:**

| 안 | 내용 | 평가 |
|---|---|---|
| A | netty event loop에 no-op 제출 후 `sync()` 배리어 | 서버측/클라측 두 군데 다 필요, `LocalChannel` peer 접근 필요. netty 내부 구현 의존 |
| **B** | 서버가 `END_SERVER_TICK`에 tick 번호 마커 패킷 송신 → 클라가 마커 N을 볼 때까지 드레인 반복 | **권장.** 스레드 토폴로지와 무관하게 결정적. 기존 `ioPhase`/`resetPhase` 시퀀스 설계와 결이 같음. **타임아웃 가드 필수** |
| C | 단순 드레인 (베스트에포트) | 2줄. 안 하는 것보단 확실히 낫지만 비결정적 |

**채택: C 먼저 → B 얹기.** C는 리스크 0으로 즉시 개선되고, B가 정확성을 완성한다.
그 전에 **측정**이 싸다 — 드레인 시점의 큐 적재 개수를 카운트해 실제 유실 빈도를 먼저 확인.

### 1.2 D5' — 왜 클라이언트 tick 고정을 없애면 안 되는가

"서버는 `TickRateManager`(simulation speed), 클라는 `DeltaTracker`(render pacing)로
분리되었으니 클라 쪽 조작은 제거해도 된다"는 판단은 **틀렸다.** 근거:

```java
// Minecraft.java:288
private final DeltaTracker.Timer deltaTracker = new DeltaTracker.Timer(20.0F, 0L, this::getTickTargetMillis);

// Minecraft.java:2988
private float getTickTargetMillis(final float defaultTickTargetMillis) {
    if (this.level != null) {
        TickRateManager manager = this.level.tickRateManager();
        if (manager.runsNormally()) {
            return Math.max(defaultTickTargetMillis, manager.millisecondsPerTick());
        }
    }
    return defaultTickTargetMillis;
}
```

`defaultTickTargetMillis`는 `msPerTick = 1000.0F / 20.0F = 50.0F`이고 `Math.max`로 **하한이
걸려 있다.** 서버 tickrate를 1000 TPS로 올려도 `Math.max(50, 1) = 50` — 클라이언트 틱
페이싱은 50ms에 바닥이 박혀 절대 빨라지지 않는다.

그리고 `ticksToDo`는 실제 벽시계 시간에서 파생된다:

```java
// DeltaTracker.Timer
public int advanceGameTime(final long currentMs) {
    this.deltaTicks = (float)(currentMs - this.lastMs) / this.targetMsptProvider.apply(this.msPerTick);
    this.lastMs = currentMs;
    this.deltaTickResidual = this.deltaTickResidual + this.deltaTicks;
    int ticks = (int)this.deltaTickResidual;
    this.deltaTickResidual -= ticks;
    return ticks;
}
```

한 RL 스텝의 벽시계 소요를 T ms라 할 때 두 가지 실패 모드가 생긴다:

- **T < 50ms** (빠른 머신): `ticksToDo == 0`인 프레임이 대부분 → `tick()`이 안 돌아
  `END_WORLD_TICK`이 발생하지 않고 동기화가 걸리지 않음. **처리량이 20 step/s에 고정**되어
  목표(20tick 해제)가 정면으로 실패.
- **T > 50ms**: `ticksToDo == 2, 3, ...` → 한 번의 render에 여러 tick →
  **액션 여러 개 소비, 관측 1개**. `1 action : 1 observation` 불변식이 깨짐.

즉 mc121의 `TickSpeedMixin`은 "억지 조작"이 아니라 **`1 action = 1 tick = 1 render =
1 observation` 불변식을 성립시키는 유일한 장치**였다.

**결론: 서버/클라 양쪽 모두 필요하다.**

- 서버: `TickRateManager.setTickRate()` / sprint → 벽시계 대기 제거
- 클라: `DeltaTracker.Timer.advanceGameTime`이 항상 정확히 1을 반환하도록 고정

부수 효과로 `deltaTickResidual = 0`이 되면 `getGameTimeDeltaPartialTick()`이 0을 반환하여
**보간 없이 정확히 tick 경계에서 렌더**된다 → 관측 결정성이 공짜로 따라온다. mc121의
`@Redirect` 꼼수보다 오히려 깨끗하다.

---

## 2. 캡처 아키텍처

### 2.1 최종 구조

```
Action
 ↓
Client input injection            (InputConstants mixin)
 ↓
Integrated Server tick            (TickRateManager로 벽시계 대기 제거)
 ↓
Tick synchronization              (TickSynchronizer + PacketProcessor 드레인)
 ↓
Client state updated
 ↓
기존 Minecraft render pipeline     (extract → render, 우리가 재실행하지 않음)
 ↓
GpuTexture (mainRenderTarget)
 ↓
 ├─ CPU capture      (glReadPixels → 추후 glGetTextureImage 검토)
 ├─ CUDA interop     (texture id → 자체 소유 FBO → CUDA IPC)
 └─ Metal IOSurface  (texture id → 자체 소유 FBO → IOSurface)
 ↓
Observation
```

### 2.2 캡처 훅 지점

`Minecraft.runTick`의 present 블록:

```java
profiler.push("present");
if (this.windowSurface.isAcquired()) {
    GpuTextureView colorTexture = this.gameRenderer.mainRenderTarget().getColorTextureView();
    ...
    this.windowSurface.blitFromTexture(RenderSystem.getDevice().createCommandEncoder(), colorTexture);
}
...
profiler.popPush("swapBuffers");
...
RenderSystem.getDevice().createCommandEncoder().submit();
if (this.windowSurface.isAcquired()) {
    this.windowSurface.present();
}
...
profiler.popPush("frameLimiter");
int framerateLimit = this.gameRenderer.gameRenderState().framerateLimit;
if (framerateLimit < 260) {
    FramerateLimiter.limitDisplayFPS(framerateLimit);
}
```

- `blitFromTexture(...)` → **redirect로 무력화** (화면 표시 스킵, 성능)
- `present()` → **redirect해서 여기서 capture + sendObservation**
  - `blitFromTexture`가 아니라 `present()`여야 한다. `submit()` **이후**여야 GPU 커맨드가
    제출된 상태이기 때문.
- `FramerateLimiter.limitDisplayFPS(...)` → **무력화 필수**. mc121엔 없던 신규 항목으로,
  present를 죽여도 이걸 안 막으면 FPS 상한에 갇힌다.

### 2.3 texture 기반 abstraction

26.2에서 `RenderTarget.fbo`(int)가 사라졌지만, GL 백엔드에서는 texture id를 여전히 얻을 수
있다:

```java
// com/mojang/blaze3d/opengl/GlTextureView.java
public GlTexture texture() { return (GlTexture)super.texture(); }
@Override public int glId() { return this.texture().id; }
```

```
RenderTarget
 └─ GpuTexture
      └─ GlTexture
           └─ GLuint texture id
```

따라서 Java↔JNI 경계는 다음처럼 바꾼다:

```
captureFramebuffer(textureId, frameBufferId, ...)   →   captureTexture(textureId, ...)
```

**단, FBO를 완전히 버리지는 않는다.** 현재 zerocopy 경로 두 개가 모두 읽기/쓰기 가능한
FBO를 전제로 하기 때문이다:

```cpp
// jni/jni_cuda_zerocopy.cpp
    jint frameBufferId, ...
    glBindFramebuffer(GL_READ_FRAMEBUFFER, frameBufferId);
    if (drawCursor) { renderCursor(mouseX, mouseY); }   // ← GPU 쪽에서 FBO에 그려 넣음
    copyFramebufferToCudaSharedMemory(targetSizeX, targetSizeY);

// jni/jni_macos_zerocopy.cpp — 동일 구조
    glBindFramebuffer(GL_READ_FRAMEBUFFER, frameBufferId);
    ...
    copyFramebufferToIOSurface(targetSizeX, targetSizeY);
```

**해결**: native가 FBO를 직접 소유한다. 최초 1회 FBO를 생성해 `GlTexture.glId()`를
`COLOR_ATTACHMENT0`에 붙여 캐시하고 매 프레임 재사용한다. 그러면
`copyFramebufferToCudaSharedMemory` / `copyFramebufferToIOSurface` / `renderCursor`는
**한 줄도 고칠 필요가 없다.** Java 쪽 추상화는 `captureTexture(textureId)`가 되고 FBO
부기(bookkeeping)만 native 내부로 숨는다.

### 2.4 glGetTextureImage 검토 (CPU path 한정)

CPU 경로는 커서를 읽기 **이후** CPU 버퍼에서 합성하므로 텍스처 직접 읽기로 바꿔도
안전하다:

```cpp
// framebuffer_capturer.cpp
GLubyte *pixels = caputreRGB(frameBufferId, textureWidth, textureHeight);
...
if (drawCursor && ...) {
    drawCursorCPU(xPos, yPos, targetSizeX, targetSizeY, pixels);   // ← CPU 합성
}
```

과거에 시도했다가 주석 처리된 흔적도 남아 있다:

```cpp
//    glBindTexture(GL_TEXTURE_2D, textureId);
//    glPixelStorei(GL_PACK_ALIGNMENT, 1);
//    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);
```

당시 안 쓴 이유 추정: (1) `glGetTextureImage`는 GL 4.5/DSA 요구 — 광범위한 GPU 호환성
고려, (2) 목표가 CPU framebuffer capture였고 GPU interop 필요성이 낮았음, (3) depth
capture까지 고려하면 FBO 방식이 자연스러움.

**우선순위는 낮다.** 2.3에서 FBO를 자체 소유하게 되면 기존 `glReadPixels` 경로가 그대로
동작하므로, `glGetTextureImage`는 **동작 확보 후 성능 최적화 단계**로 미룬다. 검증 항목:
실제 latency, 드라이버별 안정성.

---

## 3. 작업 순서

우선순위: **① tick synchronization / observation boundary → ② texture 기반 capture
abstraction → ③ render skip 최적화 → ④ TickRateManager 적용 → ⑤ 나머지 mixin**

W11~W13은 6절에서 추가된 항목으로, W1과 같은 영역을 건드리므로 **W12 → W11 → W1-b** 순서로
W1에 끼워 넣는다. W13은 독립적이라 언제 해도 무방하다.

### W1. Tick synchronization 정합성 (최우선)

- **W1-a**: `MinecraftEnv.kt`에서 `waitForServerTickCompletion()` 직후
  `client.packetProcessor().processQueuedPackets()` 호출 추가 (1.3절 C안).
- **W1-b**: tick 번호 마커 패킷으로 도착 배리어 구현 (1.3절 B안, 타임아웃 가드 필수).
  **착수 조건**: W12 계측 결과가 유의미한 유실을 보일 때. W11 이후에는
  **렌더링된 이미지 하나만** 이 배리어를 필요로 하므로, 그 전에 결정한다.

> 선행: W12(계측) → W11(서버 권위 관측) → 그 결과로 W1-b 착수 여부 판단.
- `sendObservation` 호출을 `END_WORLD_TICK`에서 **present redirect 지점으로 이동**.
  - 동기화 rendezvous 지점(`notifyServerTickStart` / `waitForServerTickCompletion` /
    `waitForClientAction` / `notifyClientSendObservation`)은 그대로 두고, capture+send만
    같은 `runTick` 내의 더 늦은 지점으로 옮긴다.
- 액션 읽기는 기존대로 `START_WORLD_TICK`(`onStartWorldTick`) 유지.
- **검증**: 액션 적용 → 관측 이미지에 그 결과가 같은 스텝에 나타나는지 (예: 시야 회전
  액션 1스텝 후 이미지의 yaw 변화 확인).

### W2. 클라이언트 tick 고정 (`TickSpeedMixin` + Accessor 재타겟)

- `@Mixin(DeltaTracker.Timer.class)` — `advanceGameTime(long)`을 HEAD에서 cancel하고 1 반환.
- Accessor(`RenderTickCounterAccessor` 계승)로 `deltaTicks`, `lastMs`,
  `deltaTickResidual` 세팅: `deltaTicks=1, lastMs=currentMs, deltaTickResidual=0`.
- **검증**: `ticksToDo`가 항상 1인지, 벽시계 부하를 걸어도 2 이상/0이 안 나오는지.

### W3. Texture 기반 capture abstraction — RGB 경로 완료

- Java: `mainRenderTarget().colorTexture` → `GlTexture` 캐스팅 → `glId()`. (`MinecraftEnv.kt` sendObservation, 완료)
- **mc121과 공유하지 않기로 결정** (mc121 영향 범위 확인 부담 제거). `captureFramebufferImpl`
  RGB 경로는 `shared-native/gl-capture/`를 쓰지 않고 `minecraft/mc262/src/main/cpp/`에 로컬
  포크(`framebuffer_capturer.cpp`, `rgb_capture.cpp`)를 두고, mc262 `CMakeLists.txt`가 이
  두 파일만 로컬 것을 쓰도록 변경. mc121 쪽은 완전히 무변경.
- 시그니처: `captureFramebufferImpl(textureId, textureWidth, textureHeight, targetSizeX,
  targetSizeY, encodingMode, isExtensionAvailable, drawCursor, xPos, yPos)` — `frameBufferId`
  파라미터 제거. native가 캡처용 FBO를 직접 소유/캐시하고, `glFramebufferTexture2D`로 주어진
  컬러 텍스처를 attach 후 `glReadPixels` (텍스처 id가 바뀔 때만 재attach — 리사이즈 등).
- 컴파일 검증: `clang++ -fsyntax-only`로 실제 헤더(jni.h, cross_gl.h, glm, png_util.h) 대상
  문법 검증 완료(HAS_PNG on/off 둘 다). **CMake/JNI 툴체인이 이 샌드박스에서 깨져있어
  실제 링크·런타임 검증은 못 함** — 다음에 정상 툴체인에서 `runClient`로 실제 프레임이
  나오는지 (검은 화면/가비지가 아닌지) 확인 필요.
- depth 경로(`captureDepthImpl`, `GameRendererDepthCaptureMixin`)는 이미 별도로 미포팅
  상태(deferred, `docs/mixin-port-todo/`)라 이번엔 손대지 않음 — 여전히 mc121과 공유.
- ZEROCOPY 경로(`captureFramebufferZerocopyImpl`/`initializeZerocopyImpl`)도 이미
  no-op/placeholder 상태였고 이번 변경 범위 밖 — 여전히 mc121과 공유, frameBufferId 기반.

### W4. GL 백엔드 fail-fast (D2)

- 초기화 시 `RenderSystem.getDevice()`가 GL 백엔드인지 확인, 아니면 즉시 예외.
- Vulkan 백엔드가 선택되면 캡처가 조용히 깨지므로 silent failure 방지가 목적.
- `Minecraft.java:479`의 `preferredGraphicsBackend.getBackendsToTry()` 참고.

### W5. Render skip 최적화 (`RenderMixin` 재설계) — 결론: blit/present no-op는 폐기

- `FramerateLimiter.limitDisplayFPS(...)` → 무력화. **이것만 적용.**
- ~~`windowSurface.blitFromTexture(...)` → no-op~~, ~~`windowSurface.present()` → no-op~~:
  **실제로 시도했다가 되돌림.** `Minecraft.doWorldLoad()`(월드 생성 시 클라이언트가 서버
  준비를 기다리는 루프)가 `while (!singleplayerServer.isReady() || gui.overlay() != null)`
  안에서 매 반복마다 `this.renderFrame(false)`를 직접 호출하는데, 여기서 present/blit를
  스킵하면 이 대기 조건이 영원히 안 풀림 — jstack으로 render thread가 이 루프 안에
  영구히 park되어 있는 것을 확인, blit/present 스킵을 제거하자 즉시 정상화되고
  `reset()`/`step()`이 실제로 끝까지 성공(SMOKE TEST PASSED, 5 step 완주)함.
  present()를 실제 캡처 훅으로 재설계(W1)하기 전까지는 건드리지 않는다.
- vsync: `windowSurface.configure(config)`의 present mode 확인은 W1에서 재검토.
- mc121의 `Tessellator.getInstance().clear()`는 대응 개념 불확실 — 필요성 재검토
- `RenderSystem.pollEvents()`는 26.2에서 public static이므로 Invoker mixin 경유 불필요

### W6. 서버 TickRateManager 적용 (D5)

- `MinecraftServer_tickspeedMixin` **삭제** (mc121 원본도 본문 전체가 주석 처리된 죽은 코드).
- 대신 초기화 시 `server.tickRateManager().setTickRate(...)` 호출.
- `TickRateManager.MIN_TICKRATE = 1.0F`로 하한만 있고 상한은 없음.
- sprint 모드(`isSprinting()` + `checkShouldSprintThisTick()`)면 `thisTickNanos = 0`이 되어
  `waitUntilNextTick`을 건너뜀 — 어느 쪽을 쓸지는 구현 시 결정.

### W7. 기계적 mixin 포팅

| mixin | 26.2 타겟 |
|---|---|
| `ServerPlayNetworkHandlerDisableSpamChecker` | `ServerGamePacketListenerImpl.detectRateSpam(TickThrottler)` HEAD cancel (chat/command 둘 다 커버) |
| `EntityCollisionDetectorMixin` | `Entity.playerTouch(Player)` |
| `AbstractBlockCollisionDetectorMixin` | `BlockBehaviour$BlockStateBase.entityInside(Level, BlockPos, Entity, InsideBlockEffectApplier, boolean)` |
| `ChatVisibleMessageAccessor` | `ChatComponent.allMessages` (`List<GuiMessage>`) |
| `GammaMixin` | `OptionInstance` (`caption`, `codec` 필드 확인됨, 메서드 시그니처 **재검증 필요**) |
| `ClientRenderInvoker` | `Minecraft.runTick(boolean)` — 다만 실사용처가 없으면 제거 검토 |

### W8. 엔티티 관측 리스너 통합

- mc121의 `ClientWorldMixin` + `WorldRendererCallEntityRenderMixin` 2개를
  `LevelExtractor` 하나로 통합.
- `extractVisibleEntities(...)` HEAD → 리스너 clear
- `extractEntity(Entity, float)` → `listener.onEntityRender(entity)`
- **semantics 개선**: 이 루프는 이미 프러스텀 컬링(`Frustum`, `isEntityVisible`)이 적용된
  뒤이므로 "실제로 화면에 보이는 엔티티"만 잡힌다. mc121 대비 관측 의미가 명확해지지만
  **결과 재현성 관점에서는 변화**이므로 인지 필요.

### W9. `InputUtilMixin` 재작성

`InputConstants`로 개명 + 모든 메서드의 첫 파라미터가 `long handle` → `Window`로 변경:

| mc121 | 26.2 |
|---|---|
| `isKeyPressed(long, int)` | `isKeyDown(Window, int)` |
| `setMouseCallbacks(long, ...)` | `setupMouseCallbacks(...)` |
| `setKeyboardCallbacks(long, ...)` | `setupKeyboardCallbacks(...)` |
| `setCursorParameters(long, int, double, double)` | `grabOrReleaseMouse(Window, int, double, double)` |

`KeyboardInfo` / `MouseInfo`의 `setHandle(long)` API도 `Window`를 받거나 `Window`에서
handle을 추출하도록 함께 수정.

### W10. 삭제

- `MinecraftServer_tickspeedMixin` (W6에서 대체)
- `WindowOffScreenMixin` (mc121 원본도 본문 전체 주석 처리된 죽은 코드,
  26.2엔 `WindowProvider.createWindow`도 없음)

### W11. 서버 권위 관측 (Seam A)

6.1절 (B). 수치 관측(position/health/food/saturation 등)을 클라 플레이어 대신 **서버
엔티티에서 직접 읽도록 이전** — 기존 `// TODO: Use server player stats instead of client
player stats` 주석의 실현. 이것만으로 수치 관측은 락의 happens-before로 완전 보장되어
배리어가 불필요해진다 (1.3절).

- `ObservationSource` 인터페이스 도입 (6.3절 Seam A), `ServerAuthoritativeSource` 구현.
- `MinecraftEnv`가 `ServerTickEvents` 콜백에서 받는 `MinecraftServer`를 보관해
  `server.playerList.players`로 `ServerPlayer` 획득.
- 이미지 관측은 이 경계 밖에 둔다 (본질적으로 클라 렌더 경로).
- **검증**: 동일 스텝에서 서버/클라 값이 어긋나는 빈도를 W12 계측과 함께 비교.

### W12. Staleness 계측 (최우선 — W1-b/W11 판단의 전제)

6.1절 (C). 기존 `CsvLogger`로 라우팅하고 printf를 쓰지 않는다.

- 드레인 시점의 `packetsToBeHandled` 적재 개수.
- 서버 tick 번호 대비 클라가 실제로 반영한 tick 번호의 차이 분포.
- 스텝당 벽시계 소요 및 그 분산.
- **산출물**: staleness 분포표. 이것이 W1-b 착수 여부와 6.1절 (C) 주장의 근거가 된다.

### W13. sync / async 모드 명시화 (Seam B)

6.1절 (C)의 정직판. 현행 `skipSync: Boolean`을 환경 옵션으로 승격한다.

- `StepBarrier` 인터페이스 도입 (6.3절 Seam B).
- `LockStepBarrier`(현행 `TickSynchronizer` 래핑) / `NoOpBarrier`(async) 2종만 구현.
  분산 배리어는 만들지 않는다 (6.4절).
- tick 이벤트 핸들러가 동기화 구현을 직접 알지 못하게 한다 — 나중에 배리어를 교체할 때
  핸들러를 건드리지 않기 위함.
- `InitialEnvironment` 프로토콜에 모드 필드 추가, 파이썬 API에 노출.

---

## 4. 범위 제외 (D4)

`RenderLayerTrianglesMixin` + `customentity/` 5개 파일(`ModelCache.kt`, `OBJLoader.kt`,
`RealisticHuman.kt`, `RealisticHumanModel.kt`, `RealisticHumanRenderer.kt`).

`Tessellator` / `RenderPhase` / `RenderLayer.MultiPhaseParameters` / `VertexConsumerProvider`
전부가 26.2 Blaze3D 재작성으로 사라졌고, `RenderPipeline` 기반으로 전면 재설계가 필요하다.
세 목표(동기화/성능/tick 해제) 어디에도 걸리지 않는 부가 기능이므로 별도 이슈로 분리.

---

## 5. 미해결 / 검증 필요

- **`GammaMixin`의 26.2 메서드 시그니처**: `caption` / `codec` 필드 존재는 확인됨.
  `getCodec()` → `codec()` 개명 여부, `setValue` 존치 여부는 미확인.
- ~~**통합 서버 패킷 flush 타이밍**~~ → **확인 완료. 보장되지 않는다.** 1.3절 참고.
- **vsync 무력화 경로**: `options.enableVsync()`를 끄는 것만으로 충분한지, 아니면
  `windowSurface.configure(...)`의 present mode까지 건드려야 하는지.
- **GPU interop 동기화**: `submit()` 후 CUDA/Metal이 텍스처를 읽을 때 fence가 필요한지.
  26.2에 `createFence()` / `RenderSystem.executePendingTasks()`가 있으므로 검토.
- **`shared-native/gl-capture/` 시그니처 변경의 mc121 영향 범위**: 구 API 유지 + 신규
  추가로 갈지, 양쪽 동시 전환할지 결정 필요.
- **`ClientRenderInvoker` 실사용처**: mc121에서 호출부가 모두 주석 처리되어 있음.
  W1 설계상 필요 없다면 W10(삭제)로 이동.

---

## 6. 향후 확장 및 ROI

1.3절에서 드러난 "관측 staleness" 문제를 계기로 스코프 확장 후보 3가지를 검토했다.
결론부터: **2개는 지금 거의 공짜로 넣고, 1개는 별도 프로젝트로 분리한다.**

### 6.1 검토한 후보와 판정

**(A) 실제 데디케이티드 서버 지원 (멀티플레이어)**

학술 가치는 실재한다 — Minecraft 규모에서 **픽셀 관측 기반 multi-agent**는 MineRL/Malmo
계열이 제대로 못 채운 영역이다. 다만 비용 구조가 "기능 추가"가 아니라 "재설계"다:

- 동기화가 **분산 배리어**가 된다. 현재는 한 JVM 안의 condvar(마이크로초). N개 클라이언트
  프로세스를 매 스텝 lockstep으로 묶으면 스텝당 최소 네트워크 RTT → **목표 2(고성능)와
  정면충돌.**
- 데디케이티드 서버는 렌더를 하지 않으므로 **에이전트마다 Minecraft 클라이언트 프로세스가
  하나씩** 필요하다. 픽셀 관측이면 GPU도 N개분.
- 아래 (B)를 **잃는다.**

→ **Phase 2 범위 밖. 6.4절로 분리.**

**(B) 서버 상태 직접 읽기**

> ⚠️ 방향이 반대다. (A)를 하면 (B)가 **가능해지는 게 아니라 불가능해진다.**

지금 서버 상태를 읽을 수 있는 **유일한 이유가 IntegratedServer**다. 같은 JVM이라 객체
참조가 그대로 닿는다:

```java
// Minecraft.java:346, 1616
private @Nullable IntegratedServer singleplayerServer;
levelName = this.getSingleplayerServer().getWorldData().getLevelName();
```

게다가 `MinecraftEnv.kt`는 이미 `ServerTickEvents` 콜백에서 `server: MinecraftServer`를
통째로 받고 있어, `server.playerList.players`로 `ServerPlayer`에 바로 닿는다.
실서버로 가면 클라는 별도 프로세스가 되어, 서버에도 모드를 깔고 별도 IPC를 뚫어야 한다.

→ **(A)와 무관하게 지금 당장 한다. 비용 거의 0.** (W1-c → W11로 승격)

**(C) "비동기 = 현실적"이라는 포지셔닝**

원칙적인 real-time/async RL 문헌이 다루는 비동기는 **경계가 있고 알려진 지연**이다
(액션이 t+Δ에 반영되고 Δ를 모델이 안다). 우리가 지금 가진 것은 스레드 스케줄링에서 나오는
**경계 없는 지터**로, "현실적"이 아니라 **"측정하지 않은 노이즈"**다. 통제하지 않은 속성을
특징으로 주장할 수는 없다.

**정직한 버전**으로 바꾸면 약점이 기여로 뒤집힌다:

1. staleness를 **계측**한다 (W12).
2. `sync` / `async`를 **명시적 환경 옵션**으로 노출한다 (W13). 이미 `skipSync` 플래그가
   있으므로 사실상 있는 것을 승격시키는 수준이다.
3. "관측 staleness를 정량화했고, 정확성이 필요하면 sync를, 처리량이 필요하면 async를
   선택한다"로 서술한다.

→ **계측과 모드 명시화를 전제로 채택.** 계측 없는 주장만 하는 것은 채택하지 않는다.

### 6.2 ROI 요약

| 항목 | 학술 가치 | 비용 | 판단 |
|---|---|---|---|
| (B) 서버 권위 관측 | 낮음 (정확성 기반) | **거의 0** | **W11 — 즉시** |
| staleness 계측 | 중 (모든 주장의 전제) | 낮음 | **W12 — 즉시** |
| sync/async 모드 명시화 = (C) 정직판 | 중~높음 | 낮음 | **W13 — 즉시** |
| 이미지 배리어 (W1-b) | 낮음 | 중 | W12 결과 보고 결정 |
| (C) 계측 없는 "현실적 비동기" 주장 | **음수** | 0 | ❌ |
| (A) 심볼릭 다중 `ServerPlayer` | 중 | 중 | 6.4절 선행 조사 |
| (A) 픽셀 관측 N프로세스 | **높음** | **매우 높음** | 별도 프로젝트 |

### 6.3 지금 남겨둘 이음새 (구조 보존 최소 설계)

목적은 **추상화를 미리 만드는 것이 아니라, 나중에 확장할 때 호출부를 갈아엎지 않도록
최소한의 경계선만 긋는 것**이다. 아래 2개면 충분하다.

**Seam A — 관측 소스 경계**

`sendObservation`이 `client.player`를 직접 참조하는 대신 한 겹을 둔다. 구현체 교체만으로
IntegratedServer / 실서버 클라이언트 양쪽을 커버한다.

```kotlin
internal interface ObservationSource {
    val x: Double; val y: Double; val z: Double
    val pitch: Float; val yaw: Float
    val health: Float; val foodLevel: Int; val saturation: Float
    val isDead: Boolean
    // ...
}

// IntegratedServer 전용 — 락의 happens-before로 완전 보장 (1.3절)
internal class ServerAuthoritativeSource(private val serverPlayer: ServerPlayer) : ObservationSource

// 실서버 클라이언트 fallback — 패킷 배달 의존
internal class ClientLocalSource(private val localPlayer: LocalPlayer) : ObservationSource
```

초기화 시 한 번 선택한다. 이미지 관측은 본질적으로 클라 렌더 경로이므로 이 경계 밖에 둔다.

**Seam B — 스텝 배리어 경계**

현재 `skipSync: Boolean` + `TickSynchronizer` 직접 호출을 인터페이스 뒤로 넣는다.
tick 이벤트 핸들러가 동기화 구현을 몰라야, 나중에 분산 배리어를 끼울 때 핸들러를 안 건드린다.

```kotlin
internal interface StepBarrier {
    fun onClientTickEnd()      // 서버에게 진행 허용 + 서버 틱 완료 대기
    fun onServerTickStart()    // 클라 액션 적용 대기
    fun onServerTickEnd()      // 클라에게 관측 시작 알림
    fun terminate()
}

internal class LockStepBarrier(...) : StepBarrier   // 현행 TickSynchronizer 래핑
internal class NoOpBarrier : StepBarrier            // async 모드 (현행 skipSync 대체)
// 향후: DistributedBarrier — 6.4절
```

**Seam C — 캡처 백엔드 경계**: W3에서 이미 texture 기반으로 정리되므로 추가 작업 없음.

### 6.4 의도적으로 하지 않을 것 (YAGNI)

과잉 설계를 막기 위해 명시한다.

- **싱글턴 해체 금지.** `FramebufferCapturer`, `MouseInfo`, `KeyboardInfo`는 Kotlin
  `object`(싱글턴)다. 이를 에이전트별로 키잉하는 작업은 **한 JVM 안에 여러 에이전트**를
  둘 때만 필요하다. 멀티플레이어의 유력 경로는 **에이전트당 1프로세스**이고, 그 모델에서는
  싱글턴이 그대로 정답이다. 지금 해체하면 순수 손해.
- **분산 배리어 구현 금지.** Seam B의 인터페이스만 긋고 구현은 만들지 않는다.
- **네트워크 프로토콜 설계 금지.** 6.5절 선행 조사 결과가 나오기 전까지 착수하지 않는다.

### 6.5 별도 프로젝트: 멀티플레이어 확장

Phase 2 완료 후 별도 문서로 스코핑한다. 두 갈래가 있고 비용 차이가 크다.

**갈래 1 — 심볼릭 다중 `ServerPlayer` (저비용 후보)**

픽셀 관측이 필요 없는 실험이라면, 하나의 IntegratedServer에 `ServerPlayer`를 여러 개 두는
방식이 훨씬 싸다. 프로세스 1개, 동기화는 기존 락 그대로, 서버 권위 관측(Seam A)도 유지된다.
대신 각 에이전트가 자기 시점 화면을 갖지 못한다.

> **선행 조사 항목**: 클라이언트 없이 `ServerPlayer`를 추가로 스폰·조작하는 것이 실제로
> 동작하는지. 동작한다면 "멀티에이전트"의 상당 부분을 1/N 비용으로 커버할 수 있는지 판단이
> 선다. **이 조사를 다른 무엇보다 먼저 한다.**

**갈래 2 — 픽셀 관측 N프로세스 (고비용)**

데디케이티드 서버 1 + 클라이언트 프로세스 N. 학술 가치가 가장 높지만
6.1절 (A)의 비용을 전부 치른다. Seam B의 `DistributedBarrier`가 여기서 필요해진다.
갈래 1 조사 결과를 보고 착수 여부를 결정한다.
