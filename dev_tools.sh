format_code() {
    # Get a list of tracked files matching the desired extensions
    git ls-files -- '*.h' '*.cpp' '*.mm' | xargs clang-format -i
    git ls-files -- '*.java' -z | xargs -0 -P 4 google-java-format -i
    ktlint '!**/com/kyhsgeekcode/minecraftenv/proto/**' --format
}
