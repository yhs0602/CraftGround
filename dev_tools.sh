format_code() {
    # Get a list of tracked files matching the desired extensions
    git ls-files -- '*.h' '*.cpp' '*.mm' | xargs clang-format -i
    git ls-files -- '*.java' -z | xargs -0 -P 4 google-java-format -i
    ktlint '!**/com/kyhsgeekcode/minecraftenv/proto/**' --format
    black .
}

generate_proto() {
    (
        set -e
        cd src/
        protoc proto/action_space.proto --python_out=craftground --pyi_out=craftground
        protoc proto/initial_environment.proto --python_out=craftground --pyi_out=craftground
        protoc proto/observation_space.proto --python_out=craftground --pyi_out=craftground
        protoc proto/action_space.proto --java_out=../shared-java/src/main/java/ --kotlin_out=../shared-java/src/main/java/
        protoc proto/initial_environment.proto --java_out=../shared-java/src/main/java/ --kotlin_out=../shared-java/src/main/java/
        protoc proto/observation_space.proto --java_out=../shared-java/src/main/java/ --kotlin_out=../shared-java/src/main/java/
    )
}

# docs/26_2_MigrationPlan.md item (g): local equivalents of the CI stages already split across
# .github/workflows/{gradle,python-ci,publish-build-runtime-packages}.yml
# (protocol-tests -> build-mod-mc121/build-mod-mc262 -> build-native-ipc -> assemble-runtime-packages),
# so a contributor can run the whole pipeline (or one stage) locally with one command instead of
# only via CI. This extends dev_tools.sh in place rather than replacing it with a separate
# scripts/build-runtime.py orchestrator, since format_code/generate_proto above already live here.

# Stage 1: protocol-tests - proto codegen must be current, then the Python unit tests that exercise
# the wire format (including the HandshakeAck handshake added for item (f)) must pass.
protocol_tests() {
    generate_proto || return 1
    PYTHONPATH=src:./build python -m pytest tests/python/unit/
}

# Stage 2: build-mod-mc121 / build-mod-mc262 - each Minecraft version is its own Gradle root
# (docs/26_2_MigrationPlan.md's "완료됨" section); mirrors gradle.yml's gradle_build/gradle_build_mc262 jobs.
build_mod() {
    local mc_dir="$1" # mc121 or mc262
    (cd "minecraft/${mc_dir}" && ./gradlew build)
}

build_mods() {
    build_mod mc121 && build_mod mc262
}

# Stage 3: build-native-ipc / build-capture-native - the shared C++ IPC + pybind11 extension at the
# repo root; mirrors python-ci.yml's "build cpp extension" step. Each mod's own native frame-capture
# code (minecraft/*/src/main/cpp) is already built as part of build_mod above via its Gradle CMake task.
build_native_ipc() {
    mkdir -p build
    (cd build && cmake .. && cmake --build .)
}

# Stage 4: assemble-runtime-packages - sdist/wheel for each craftground-runtime-mc* package; mirrors
# publish-build-runtime-packages.yml's mc_dir matrix.
assemble_runtime_packages() {
    for mc_dir in minecraft/mc121 minecraft/mc262; do
        (cd "${mc_dir}" && python -m build) || return 1
    done
}

build_all() {
    protocol_tests \
        && build_mods \
        && build_native_ipc \
        && assemble_runtime_packages
}