### Adding a new paramerter
1. Edit .proto
2. protoc
3. Edit python files


# Formatting
## Install formatters
```zsh
brew install ktlint clang-format google-java-format
```
```bash
wget https://apt.llvm.org/llvm.sh
sudo ./llvm.sh 19
sudo apt install clang-format-19
sudo ln -s /usr/bin/clang-format-19 /usr/bin/clang-format
```

## Run formatters

```bash
git ls-files -- '*.h' '*.cpp' '*.mm' | xargs clang-format -i
git ls-files -- '*.java' -z | xargs -0 -P 4 google-java-format -i
ktlint '!**/com/kyhsgeekcode/minecraftenv/proto/**' --format
```

# Managing proto files
```bash
cd src/
protoc proto/action_space.proto --python_out=craftground
protoc proto/initial_environment.proto --python_out=craftground
protoc proto/observation_space.proto --python_out=craftground
protoc proto/action_space.proto --java_out=craftground/MinecraftEnv/src/main/java/ --kotlin_out=craftground/MinecraftEnv/src/main/java/
protoc proto/initial_environment.proto --java_out=craftground/MinecraftEnv/src/main/java/ --kotlin_out=craftground/MinecraftEnv/src/main/java/
protoc proto/observation_space.proto --java_out=craftground/MinecraftEnv/src/main/java/ --kotlin_out=craftground/MinecraftEnv/src/main/java/
```

# Troubleshooting
## Protobuf runtime version error
> google.protobuf.runtime_version.VersionError: Detected incompatible Protobuf Gencode/Runtime versions when loading proto/initial_environment.proto: gencode 5.29.1 runtime 5.27.3. Runtime version cannot be older than the linked gencode version. See Protobuf version guarantees at https://protobuf.dev/support/cross-version-runtime-guarantee.

### Solution
```bash
pip install --upgrade protobuf
```


# Dev setup & build (conda, linux)
```
conda create --name craftground python=3.11
conda activate craftground
conda install gymnasium Pillow numpy protobuf typing_extensions psutil pytorch ninja build cmake
conda install -c conda-forge openjdk=21 libgl-devel
conda install glew
python -m build
```

## Build Gradle with c++ altogether
```bash
cd src/craftground/MinecraftEnv
./gradlew build
```

## Build only jvm's c++ part
```bash
 cmake src/main/cpp -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
 cmake --build .
```

## Python unit test with coverage
```bash
python -m pip install coverage pytest
cd build
cmake ..
cmake --build .
cd ..
ln -s build/*.dylib craftground/src/
ln -s build/*.so craftground/src/
ln -s build/*.pyd craftground/src/
PYTHONPATH=./build:src coverage run --source=src/craftground -m pytest tests/python/unit/
coverage report
```