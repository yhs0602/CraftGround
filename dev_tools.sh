format_code() {
    # Get a list of tracked files matching the desired extensions
    git ls-files -- '*.h' '*.cpp' '*.mm' | xargs clang-format -i
    git ls-files -- '*.java' -z | xargs -0 -P 4 google-java-format -i
    ktlint '!**/com/kyhsgeekcode/minecraftenv/proto/**' --format
    black .
}

generate_proto() {
    cd src/
    protoc proto/action_space.proto --python_out=craftground --pyi_out=craftground
    protoc proto/initial_environment.proto --python_out=craftground --pyi_out=craftground
    protoc proto/observation_space.proto --python_out=craftground --pyi_out=craftground
    protoc proto/action_space.proto --java_out=craftground/MinecraftEnv/src/main/java/ --kotlin_out=craftground/MinecraftEnv/src/main/java/
    protoc proto/initial_environment.proto --java_out=craftground/MinecraftEnv/src/main/java/ --kotlin_out=craftground/MinecraftEnv/src/main/java/
    protoc proto/observation_space.proto --java_out=craftground/MinecraftEnv/src/main/java/ --kotlin_out=craftground/MinecraftEnv/src/main/java/
}