# 26.2 Vulkan 캡처 백엔드

Minecraft 26.2는 OpenGL과 Vulkan을 모두 렌더링 백엔드로 지원한다. Phase 2까지의 mc262 캡처
경로는 `GlTexture.glId()` → 네이티브 `glReadPixels`에 완전히 묶여 있어서, Vulkan으로 뜨면
프레임을 한 장도 얻지 못했다. 이 문서는 그 제약을 없앤 **backend-neutral 캡처 경로**를
설명한다.

## 핵심 결론

**Vulkan readback을 위해 네이티브 Vulkan 코드를 한 줄도 쓰지 않았다.**

`docs/26_2_MigrationPlan.md` §6은 "`gl*`을 `vk*`로 치환하면 안 된다 — layout transition,
staging copy, barrier, fence, mapped host-visible memory가 필요하다"고 지적했다. 그 지적은
여전히 맞다. 다만 **그 일을 Blaze3D가 이미 다 해놓았다**: 26.2의 `CommandEncoder`는
백엔드 중립 GPU→CPU readback API를 노출하고, `VulkanCommandEncoder`가 그 안에서
`vkCmdCopyImageToBuffer` + barrier를, `GlCommandEncoder`가 PBO + `glReadPixels`를 각각
구현한다.

따라서 `FrameCaptureBackend` 같은 새 인터페이스 계층도 필요 없었다. 기존의
"분기만 쓰고 다형성은 안 쓴다" 원칙 그대로, 캡처 지점에서 두 갈래로 나뉜다.

## 디컴파일로 확정한 사실

| 사실 | 근거 |
|---|---|
| `CommandEncoder.copyTextureToBuffer(GpuTexture, GpuBuffer, offset, Runnable, mipLevel[, x,y,w,h])` 존재 | `com/mojang/blaze3d/systems/CommandEncoder.java:582,590` |
| 목적지 버퍼는 **tightly packed** (`w*h*format.blockSize()`) — row padding 없음 | 같은 파일 :606 크기 검증 |
| `GpuBuffer.map(read, write)` → `GpuBufferSlice.MappedView.data(): ByteBuffer` (direct) | `GpuBufferSlice$MappedView` |
| `USAGE_MAP_READ=1`, `USAGE_COPY_DST=8`. Vulkan은 MAP_READ에 host-visible+coherent를 강제 | `VulkanGpuBuffer$Direct` 생성자 |
| MainTarget color = `RGBA8_UNORM`, usage `15` = COPY_DST\|**COPY_SRC**\|TEXTURE_BINDING\|RENDER_ATTACHMENT → 복사 소스로 바로 사용 가능 | `MainTarget.java:16,74` |
| depth = `D32_FLOAT`, 동일 usage 15 | `MainTarget.java:82` |
| Vulkan 구현은 `vkCmdCopyImageToBuffer` + `memoryBarrier`. **layout transition 불필요** (MC가 이미지를 GENERAL 레이아웃으로 유지) | `VulkanCommandEncoder.java:733-777` |
| `copyTextureToBuffer`의 `Runnable callback`은 **즉시 실행되지 않는다** — `queueForDestroy`로 2프레임 뒤 실행 → 동기 readback에 쓸 수 없다 | 같은 파일 :776 |
| GL/Vulkan fence 모두 `awaitCompletion(timeoutNS)` — **나노초** 단위 | `GlFence.java`, `VulkanCommandEncoder$1.java` |

### fence 순서가 load-bearing한 이유

`GpuFence`는 **생성 시점의 submit index를 캡처**하고, `awaitCompletion()`은 그 인덱스가
제출되고 완료될 때까지 블록한다.

```java
// VulkanCommandEncoder$1
VulkanCommandEncoder$1(VulkanCommandEncoder this$0) {
    this.submitIndex = this.this$0.currentSubmitIndex;   // <-- 생성 시점
}
public boolean awaitCompletion(long timeoutMs) {
    this.completed = this.this$0.awaitSubmitCompletion(this.submitIndex, timeoutMs);
}
```

⇒ **`submit()` 이후에 만든 fence는 아직 제출되지 않은 N+1을 기다리며 영원히 멈춘다.**
그래서 복사 기록과 fence 무장은 반드시 그 프레임의 `submit()` **이전**이어야 한다.

실제 프레임 흐름:

```
GameRenderer.render()
  └ renderLevel() 직후  ── GameRendererDepthCaptureMixin
        BLAZE3D: Blaze3dCapture.recordDepthReadback()  ← 복사 "기록"만
                 (여기서 읽을 수 없다: fence를 만들면 지금 쓰고 있는 그 submission을
                  가리키게 되고, 그건 아직 제출 전이다. 그렇다고 미룰 수도 없다 —
                  26.2는 GUI 패스 전에 depth를 clear한다.)
  ...
CommandEncoder.submit()
  ├ @At BEFORE : MinecraftEnv.onBeforeSubmitCapture()
  │                → recordColorReadback() + armFence()      [submit index N]
  ├ (vanilla)  : N 제출
  └ @At AFTER  : MinecraftEnv.onPresentCapture()
                   → awaitPendingFence()  (N 완료 대기)
                   → readColor() / readPendingDepth()  = map + 변환
```

**추가 submit이 없다.** GL 백엔드에서도 동일하게 유효하다.

**stereo만 예외**: `renderEyeAndCapture()`는 프레임 중간에 재렌더하고 즉시 픽셀이 필요한데,
바닐라는 그 프레임을 제출할 생각이 없다. 그래서 `Blaze3dCapture.captureNow()`가
`record → armFence → submit() → await → map`을 눈마다 한 번씩 직접 수행한다 (눈당 submit 1회).

### 상하 반전 (flip)

**둘 다 `flipVertically = false`다.** 애초에는 Vulkan 쪽에 `true`가 필요하다고 추론했었다:

- 26.2 Vulkan 백엔드는 `VulkanRenderPass`에서 평범한 양수 height 뷰포트를 설정한다
  (negative-height viewport 트릭을 쓰지 않는다).
- 대신 `VulkanRenderPipeline`이 모든 파이프라인에 `frontFace(1)` = `VK_FRONT_FACE_CLOCKWISE`를
  선언한다. GL 기하는 CCW가 정면인데, Y축이 뒤집혔을 때만 필요한 winding 보정이 이것이라 추론했다.
- ⇒ Vulkan은 y-down NDC로 렌더하니 이미지 row 0이 화면 위쪽이고, `glReadPixels`의 첫 row(화면
  아래쪽)와 정확히 뒤집혀 있을 거라 예상했다.

**Apple Silicon(MoltenVK)에서 실제로 캡처해보니 이 추론은 틀렸다**: `minecraft-simulator-benchmark`
하네스로 Vulkan RAW 프레임을 떠 보면(§3 참고) `flipVertically = true`(기존 기본값)일 때 상하가
뒤집혀 나오고, `-Dcraftground.blaze3dFlipVertically=false`로 강제하면 GL 베이스라인과 동일하게
나온다 — 즉 이 하드웨어에서는 `vkCmdCopyImageToBuffer`가 이미 `glReadPixels`와 같은 row 순서로
돌려준다. winding 보정과 readback row 순서는 서로 무관했던 것. 기본값을 `false`로 고쳤다.

**중요 — 이건 Apple Silicon/MoltenVK 한 대에서만 검증했다.** 뷰포트/winding 설정은 플랫폼 불문
동일한 Java/LWJGL 코드라서 Linux/Windows 네이티브 Vulkan 드라이버(AMD/NVIDIA/Intel)에서도
`false`가 맞을 개연성은 있지만, 실제로 확인된 사실은 아니다 — 감이나 스펙 문서가 아니라 딱 이
한 대에서의 바이트 단위 비교 결과일 뿐이다. 다른 플랫폼에 배포하기 전에 같은 §3 비교를 그
하드웨어에서 반드시 재현해 보고, 다시 어긋나면 `-Dcraftground.blaze3dFlipVertically=<bool>`로
뒤집을 수 있다.

## 구현 지도

| 파일 | 역할 |
|---|---|
| `minecraft/mc262/src/client/java/.../Blaze3dCapture.kt` | **신규.** backend 판별, readback 버퍼 캐시, record/arm/await/read, depth 선형화, stereo용 `captureNow`. Blaze3D 타입이 client-only라 client 소스셋에 있다 |
| `.../mixin/RenderMixin.java` | `submit()`에 `@At BEFORE` 훅 추가 (기존 AFTER 훅 유지) |
| `.../MinecraftEnv.kt` | `GlTexture` 하드 캐스트 제거, `onBeforeSubmitCapture` 배선, 백엔드별 분기, depth 지연 읽기 |
| `.../mixin/GameRendererDepthCaptureMixin.java` | BLAZE3D면 복사 기록만, GL이면 기존 `captureDepthImpl` |
| `.../EnvironmentInitializer.kt` | `checkGlBackend` → `checkRenderBackend`. 두 백엔드 모두 허용 + 캡처 경로 로그. Vulkan+ZEROCOPY만 fail-fast |
| `src/main/java/.../FramebufferCapturer.kt` | `VULKAN=3` 상수 제거, `convertCapturedFrameImpl` 선언 추가. GL 경로는 그대로 |
| `src/main/cpp/frame_convert.cpp` + `include/frame_convert.h` | **신규.** RGBA→RGB(+flip) 변환과, 두 경로가 공유하는 `makeByteStringFromRgb` (resize/cursor/PNG/ByteString) |
| `src/main/cpp/framebuffer_capturer.cpp` | 꼬리 로직을 위 헬퍼로 추출해 축소 |

`find_package(Vulkan)`은 없다. Vulkan readback이 전부 Kotlin 쪽 Blaze3D 호출이기 때문이다.

Python 쪽은 **변경 없음**. wire format이 동일하고 백엔드는 Java 런타임에서 자동 감지된다.

### 인코딩 모드와 백엔드는 직교한다

이전 스캐폴딩에는 `FramebufferCapturer.VULKAN = 3`이라는 encoding mode 상수가 있었다. 제거했다.

- Vulkan으로 돌려도 결과는 여전히 RAW 또는 PNG다. 백엔드는 인코딩이 아니다.
- 게다가 ordinal 3은 Python `ScreenEncodingMode.ZEROCOPY_JAX = 3`과 충돌했다 —
  JAX 모드를 mc262에서 쓰면 mc262가 그걸 VULKAN으로 읽고 예외를 던졌을 것이다.

## 설정

| 설정 | 기본값 | 용도 |
|---|---|---|
| `CRAFTGROUND_CAPTURE_BACKEND` 환경변수 (또는 `-Dcraftground.captureBackend`) = `opengl`\|`blaze3d` | 자동 감지 | 캡처 경로 강제. GL 머신에서 `blaze3d`를 켜서 두 경로를 바이트 단위로 비교하는 용도 |
| `-Dcraftground.blaze3dFlipVertically=<bool>` | false (Apple Silicon/MoltenVK에서 검증됨, 다른 플랫폼 미검증) | 위 flip 기본값 무효화 |
| `-Dcraftground.enableMetalObjects=true` | false | Vulkan 디바이스 생성 시 물리 디바이스가 지원하면 `VK_EXT_metal_objects`를 활성화 (ZEROCOPY (Metal) 선행 작업, 확장 활성화까지만 — 이미지 import는 미구현) |

## 검증

### 1. 빌드

```bash
cd minecraft/mc262 && ./gradlew build
```

(통과 확인됨. `convertCapturedFrameImpl` 심볼이 `libnative-lib.dylib`에 들어간 것도 확인.)

### 2. GL 경로 회귀 없음

`CRAFTGROUND_CAPTURE_BACKEND`를 설정하지 않고 기존 스모크 테스트를 돌린다. GL 머신에서는
자동 감지가 `OPENGL`을 고르므로 캡처 코드 경로가 이전과 **완전히 동일**해야 한다
(`glReadPixels` → `makeByteStringFromRgb`, 리팩터링으로 함수만 옮겼을 뿐 로직 동일).
RAW / PNG / depth / stereo 네 가지를 모두 확인.

### 3. 픽셀 동일성 — flip과 채널 순서를 확정하는 단계

같은 GL 머신에서 같은 시드·같은 액션 시퀀스로 두 번 돌려 프레임을 덤프하고 비교한다.

```bash
CRAFTGROUND_CAPTURE_BACKEND=opengl  python <smoke_script>   # baseline
CRAFTGROUND_CAPTURE_BACKEND=blaze3d python <smoke_script>   # 신규 경로
```

두 덤프는 **바이트 단위로 일치**해야 한다 (GL 디바이스에서는 flip이 false이고 두 경로 모두
`glReadPixels` 결과를 쓰므로 구조적으로 같아야 한다). 불일치하면:

- 상하 반전 → `blaze3dFlipVertically()` 로직
- R/B 뒤바뀜 → `frame_convert.cpp`의 채널 순서 (`GlConst.toGlExternalId(RGBA8_UNORM)`가 RGBA인지 확인)
- 가장자리만 다름 → `resize_pixels` 경계

이 단계가 통과해야 Vulkan 결과를 신뢰할 수 있다.

### 4. Vulkan 실행

`options.txt`의 graphics backend를 Vulkan으로 두거나 GL 백엔드 생성을 실패시켜 26.2가
Vulkan을 고르게 한 뒤, `EnvironmentInitializer`가 찍는 로그로 확인한다:

```
CraftGround: rendering backend 'Vulkan' -> capture path BLAZE3D (color texture VulkanGpuTexture)
```

확인 항목:

- RAW 단안 → GL 결과와 육안/수치 일치 (§3의 baseline 덤프와 비교)
- depth → 근/원거리 값이 GL depth와 일치 (reverse-Z 선형화가 Kotlin으로 포팅된 것 검증)
- stereo (`eyeDistance > 0`) → 좌/우 프레임이 다르고 시차 방향이 올바름
- PNG 모드
- Vulkan + `ZEROCOPY_TORCH` → 초기화 시점에 명확한 예외 (조용히 깨지지 않음)

### 5. 레이턴시

`csvLogger.profile*`가 이미 `.../SendObservation/Prepare/SingleEye/ByteString` 구간을 찍는다
(`CRAFTGROUND_JAVA_PROFILE=1`). GL 네이티브 경로 대비 Blaze3D 경로의 스텝 시간을 비교해
기록한다. 유의미하게 느리면(>2x) 네이티브 Vulkan JNI 경로를 후속 작업으로 올리고, 그 판단
근거를 여기에 남긴다.

---

## ZEROCOPY (Metal) — GL 포팅 + 확장 활성화 완료, 이미지 import는 아직

### GL ZEROCOPY (텍스처 기반)

mc262 GL ZEROCOPY가 포팅됐다. mc121은 FBO 정수를 `glCopyTexSubImage2D`에 넘겼지만, mc262는
RAW/PNG와 마찬가지로 FBO가 없다. `glCopyImageSubData`(GL 4.3 core)도 고려했지만 macOS의
`OpenGL.framework`는 GL 4.1 core profile까지만 지원해 그 심볼을 안전하게 믿을 수 없어서, 대신
매 프레임 소스 텍스처를 임시 FBO에 붙이고 `glBlitFramebuffer`(GL 3.0 core, 서로 다른 텍스처
타겟 간에도 동작)로 IOSurface 백업 `GL_TEXTURE_RECTANGLE_ARB` 텍스처에 복사한다
(`(colorTexture as GlTexture).glId()` → `targetFbo`). 에이전트용 커서 오버레이는 그 대상 FBO에
이어서 그린다 — mc121처럼 실제 화면 프레임버퍼에 그리지 않으므로, 사람이 보는 게임 창에는
나타나지 않는다(RAW 경로의
`drawCursorCPU`가 CPU 버퍼에만 그리는 것과 같은 원칙).

여전히 OpenGL 백엔드 전용이다 — `EnvironmentInitializer.checkRenderBackend`가 Vulkan +
`ZEROCOPY_TORCH` 조합을 그대로 fail-fast 처리한다. Depth ZEROCOPY는 mc121에도 없어서
(`TODO: Depth buffer`) 포팅하지 않았다.

관련 파일: `shared-native/gl-capture/framebuffer_capturer_apple.mm` (+cuda/dummy 변형),
`jni/jni_macos_zerocopy.cpp`, `FramebufferCapturer.kt`의 `initializeZeroCopy`/
`captureFramebufferZerocopyImpl`, `MinecraftEnv.kt`의 `sendObservation`(초기화 1회 호출 +
`ipcHandle` 필드 채우기).

### Vulkan + `VK_EXT_metal_objects`: 확장 활성화까지 완료

`VulkanBackendMetalExtensionMixin` (client mixin)이 `VulkanBackend`의 private
`createDevice(Collection<String>, VulkanPhysicalDevice, Set<VulkanFeature>)` 호출을
`@Redirect`로 가로채, 물리 디바이스가 `VK_EXT_metal_objects`를 지원하면 그 확장을
`deviceExtensions`에 추가한 뒤 원래 메서드를 그대로 호출한다. `@Shadow`로 그 private 메서드를
직접 호출한다(별도 accessor/invoker 불필요 — mixin이 같은 클래스로 병합되므로).

**opt-in**: `-Dcraftground.enableMetalObjects=true`가 없으면 이 mixin은 아무것도 바꾸지 않는다
(기존 `deviceExtensions`를 그대로 넘김). Mojang의 디바이스 생성 경로에 개입하는 가장 위험도
높은 지점이라, 검증 전까지 기본 Vulkan 실행에 영향을 주지 않도록 게이팅했다.

**아직 안 한 것**: 확장이 활성화된 이후의 실제 사용 — IOSurface 백업 `MTLTexture`를
`VkImportMetalTextureInfoEXT`로 `VkImage`로 import하고, 매 프레임 `vkCmdCopyImage`로 MC의
color image를 그리로 복사하는 것. 이건 네이티브 Vulkan 코드가 필요한 별도 작업이고, LWJGL에
`VK_EXT_metal_objects` 바인딩이 있는지부터 확인해야 한다. mach port를 Python에 넘기는 마지막
단계는 기존 경로(`createMachPortForIOSurface` / `observation_converter.py:254`의
`initialize_from_mach_port`)를 그대로 재사용할 수 있을 것으로 보인다 — Python 쪽은 IOSurface가
어느 API로 채워졌는지 신경 쓰지 않기 때문이다.

### 걸림돌이었던 사실 (여전히 유효): MC가 `VK_EXT_metal_objects`를 기본으로 켜지 않는다

`VulkanBackend.java:59`의 필수 확장 목록은 이게 전부다:

```
VK_KHR_dynamic_rendering, VK_KHR_push_descriptor, VK_KHR_synchronization2,
VK_EXT_vertex_attribute_divisor, VK_KHR_swapchain
```

여기에 조건부로 `VK_KHR_portability_subset`(macOS), `VK_AMD_buffer_marker` /
`VK_NV_device_diagnostic_checkpoints`, `VK_EXT_multi_draw`가 붙는다
(`VulkanBackend.java:147-160`).

`VK_EXT_metal_objects`가 없으므로 `vkExportMetalObjectsEXT`도, 이미지 생성 시
`VkImportMetalTextureInfoEXT`도 쓸 수 없다. **디바이스 생성 시점에 활성화돼 있어야 하는
확장**이라 사후에 끼워넣을 수 없다.

### 남은 구현 스케치

1. ~~`VulkanBackend`의 `deviceExtensions` 리스트에 `VK_EXT_metal_objects`를 주입하는 mixin~~ —
   완료 (`VulkanBackendMetalExtensionMixin`, opt-in).
2. IOSurface 백업 MTLTexture를 만들고 (`shared-native/gl-capture/framebuffer_capturer_apple.mm`의
   `createSharedIOSurface`가 이미 한다), 그걸 `VkImage`로 import.
3. 매 프레임 `vkCmdCopyImage`로 MC의 color image → 우리 이미지.
4. mach port를 Python으로 전달 — **기존 경로 그대로**
   (`framebuffer_capturer_apple.mm`의 `createMachPortForIOSurface`,
   `src/craftground/environment/observation_converter.py:254`의 `initialize_from_mach_port`).

**Python 쪽은 변경이 필요 없다.** 지금도 `ipc_handle` 바이트를 mach port로만 해석하고,
IOSurface가 어느 API로 채워졌는지는 신경 쓰지 않는다.

### 남은 선행 작업 순서

1. ~~mc262 GL ZEROCOPY 포팅 (W3) — texture 기반으로 IOSurface 채우기~~ — 완료.
2. ~~Vulkan + `VK_EXT_metal_objects` mixin~~ — 완료 (opt-in, 기본 off).
3. 위 2~4단계 (이미지 import, `vkCmdCopyImage`, mach port 전달) — 미착수.
