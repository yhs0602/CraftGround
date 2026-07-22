# 26.2 마이그레이션 — Phase 1 작업 문서 (빌드 구조 재정비)

`26_2_MigrationPlan.md`, `26_2_best_strategy.md`의 결론을 현재 레포 상태에 맞춰
구체적인 이동/수정 목록으로 정리한다. **범위는 Phase 1(구조 재정비 + 빌드/CI 안정화)까지이며,
mc262 실구현은 Phase 2로 미룬다.**

## 0. 현재 상태 요약

- `src/craftground/MinecraftEnv/` — Fabric mod(mc1.21) 전체. 자체 `gradlew`, Loom `1.6-SNAPSHOT`,
  Gradle 8.8. Java(mixin) + **자체 native cpp**(`src/main/cpp`, JNI/OpenGL 캡처, `configureCppProject`
  Gradle task가 CMake로 빌드)를 포함.
- `src/cpp/` — Python↔JVM IPC (pybind11, boost/noboost, cuda). 최상위 `CMakeLists.txt` +
  `pyproject.toml`(scikit-build-core)로 빌드되어 파이썬 wheel에 native extension으로 들어감.
  → 전략 문서가 말하는 "공통 가능" 계층에 해당.
- `src/proto/*.proto` — 파이썬 쪽만 codegen되어 `src/craftground/proto/*_pb2.py`로 존재. Java 쪽은
  손으로 짠 클래스(`MinecraftEnv/.../proto/*.java`)라 진짜 공유 스키마는 아직 없음.
- `src/craftground/` — 파이썬 API 패키지 본체(gymnasium env, wrappers 등). scikit-build-core의
  src-layout으로 이미 패키징되어 있음.
- mc262 프로젝트는 아직 없음 (디렉토리 자체가 존재하지 않음).

## 1. 목표 디렉토리 구조 (Phase 1 기준)

권장안의 `python/`을 그대로 쓰지 않고, 기존 scikit-build-core src-layout을 유지한다
(이유는 2절 "의도적으로 다르게 가는 부분" 참고).

```
CraftGround/
├── settings.gradle.kts        # IDE 전용 composite build (신규)
├── minecraft/
│   └── mc121/                 # src/craftground/MinecraftEnv/ 를 git mv
│       ├── gradlew, gradle/, settings.gradle, build.gradle
│       └── src/main/{java,cpp,resources}
│   # mc262/는 Phase 2에서 추가 (지금은 만들지 않음)
├── shared/
│   └── native-ipc/            # src/cpp/ 를 git mv (Python↔JVM IPC, 공통)
├── src/
│   ├── craftground/           # 파이썬 API — 그대로 유지
│   └── proto/                 # 그대로 유지 (아직 공유 스키마 아님)
├── CMakeLists.txt             # shared/native-ipc 경로로 수정
├── pyproject.toml
└── scripts/
    └── build_all.py           # (신규) mc121 gradle build + pip build 오케스트레이션
```

`build-logic/`(공통 Gradle convention)과 `shared/protocol-java/`는 **Gradle root가 2개 이상일 때**
의미가 생기므로 mc262 스캐폴딩을 시작하는 Phase 2 초입에 추가한다. 지금 만들면 공유할 대상이
mc121 하나뿐이라 빈 뼈대만 남는다.

## 2. 원안과 의도적으로 다르게 가는 부분

- **`python/` 최상위 디렉토리로 옮기지 않음.** `pyproject.toml`이 scikit-build-core로
  `src/craftground`를 패키지 루트로 이미 쓰고 있고, `MANIFEST.in`(`graft src`), `.clangd`,
  CI의 여러 경로가 `src/`를 전제로 함. 이름만 다를 뿐 src-layout 자체는 원안의 `python/`과
  동일한 역할이므로, 이동에 따르는 리스크 대비 얻는 게 없다. **이동하지 않는다.**
- **`shared/protocol-java/`는 지금 만들지 않음.** 실제로 재사용할 Java 코드가 아직 없다
  (mc262가 없으므로). mc262 스캐폴딩 시점에 mc121에서 공유 가능한 클래스를 뽑아내며 함께 만든다.
- **`shared/native-ipc/`는 지금 만듦.** `src/cpp`는 이미 Minecraft 버전과 무관한 순수 IPC 코드라
  이동 자체는 안전하고, 이름을 명확히 해두는 게 Phase 2 진입 시 유리하다.

## 3. 이동 작업 (git mv, 히스토리 보존)

1. `git mv src/craftground/MinecraftEnv minecraft/mc121`
2. `git mv src/cpp shared/native-ipc`
3. 루트에 `settings.gradle.kts` 신규 생성 (IDE composite, `includeBuild("minecraft/mc121")`)

## 4. 경로 참조 업데이트 체크리스트

`MinecraftEnv` 문자열을 grep한 실제 참조 지점 (mc121 자체 내부 파일 제외):

| 파일 | 현재 | 변경 |
|---|---|---|
| `pyproject.toml:73,88` | `src/craftground/MinecraftEnv` (sdist 경로 나열 + exclude) | `minecraft/mc121` |
| `CMakeLists.txt` | `src/cpp` 참조 여부 확인 후 수정 | `shared/native-ipc` |
| `.clangd:10,13` | `PathMatch`/`CompilationDatabase`가 `src/craftground/MinecraftEnv/...` | `minecraft/mc121/...` |
| `.clang-format-ignore:20` | `src/craftground/MinecraftEnv/_deps/*` | `minecraft/mc121/_deps/*` |
| `.vscode/settings.json:2` | `${workspaceFolder}/src/MinecraftEnv/compile_commands.json` (이미 오탈자로 `src/craftground` 누락 — 겸사겸사 수정) | `${workspaceFolder}/minecraft/mc121/compile_commands.json` |
| `.github/dependabot.yml:9` | `directory: "/src/craftground/MinecraftEnv"` | `/minecraft/mc121` |
| `.github/workflows/gradle.yml` | `working-directory: src/craftground/MinecraftEnv`, cache path, `build-root-directory` | `minecraft/mc121` |
| `.github/workflows/ktlint.yml:17` | ktlint exclude 패턴 | `!minecraft/mc121/src/main/java/.../proto/**` |
| `.github/workflows/publish-package-cuda-linux.yml:48-49` | `_deps` 캐시 경로 + hashFiles | `minecraft/mc121/...` |
| `.github/workflows/publish-package-cuda-windows.yml:63-64` | 동일 | 동일 |
| `.github/workflows/publish-package-nocuda.yml:55-56` | 동일 | 동일 |
| `src/craftground/environment/environment.py:109,464` | mod 빌드/실행 디렉토리를 `"MinecraftEnv"` 하드코딩으로 찾음 | `"minecraft/mc121"` — **mc262 도입 시 여기가 `mc_version` 분기점이 됨** |
| `ARCHITECTURE_FLOWCHART.md`, `docs/blog/technical_report.md`, `docs/develop.md` | 경로 언급 | 문서 갱신 (기능에는 영향 없음, 후순위) |

`.github/workflows/cmake-build.yml`, `cmake-build-cuda.yml`, `java-format.yml`, `python-ci.yml`은
grep 결과에 없었지만 `src/cpp` 이동의 영향을 받을 수 있으므로 실제 수정 시 재확인 필요.

## 5. 오케스트레이션 스크립트

`scripts/build_all.py` (신규): mc121 gradle build를 파이썬 wheel 빌드와 조율.

```python
subprocess.run(["./gradlew", "build"], cwd="minecraft/mc121", check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
```

지금은 mc121 하나뿐이라 orchestrator가 하는 일이 크지 않지만, mc262가 추가되면
`--mc-version` 인자로 대상 Gradle root를 선택하는 지점이 된다. 인터페이스만 지금 잡아둔다.

## 6. 검증 순서

1. `git mv` 이후 `./minecraft/mc121/gradlew -p minecraft/mc121 build` 단독 성공 확인
2. `pip install -e .` (scikit-build-core가 `shared/native-ipc` + 최상위 `CMakeLists.txt`로 빌드) 성공 확인
3. `pytest tests/python`, `tests/cpp`(CMakeLists 기준) 로컬 통과 확인
4. 실제 env 실행 (`environment.py`가 갱신된 경로로 mod jar를 찾아 게임을 띄우는지) 스모크 테스트
5. GitHub Actions 4개 워크플로(`gradle.yml`, `cmake-build*.yml`, `publish-*.yml`) 그린 확인

## 7. 여기서 하지 않는 것 (Phase 2로 이연)

- `minecraft/mc262/` 실제 생성, Loom 1.17 / Gradle 9.5.1 wrapper 세팅
- `shared/protocol-java/` 및 `build-logic/` convention plugin
- `craftground-runtime-mc121` / `craftground-runtime-mc262` PyPI 패키지 분리
- Vulkan/Blaze3D 프레임 캡처 추상화 (`FrameCaptureBackend`)
- protocol handshake(`protocol_version` 등) 도입

---

이 문서대로 진행해도 괜찮으면, 4절 체크리스트 순서대로 `git mv` + 경로 치환을 실행하고
각 단계마다 로컬 빌드로 검증하는 식으로 진행하려 함. mc262는 Phase 1 완료 후 별도로 기획.
