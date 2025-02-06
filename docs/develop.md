---
title: Development Guide
nav_order: 7
---

# **CraftGround Development Guide**  

This guide provides instructions for setting up the **CraftGround** development environment, modifying `.proto` files, formatting the code, building the project, and running tests.  

---

## **Table of Contents**  
- [**CraftGround Development Guide**](#craftground-development-guide)
  - [**Table of Contents**](#table-of-contents)
  - [**Adding a New Parameter**](#adding-a-new-parameter)
  - [**Code Formatting**](#code-formatting)
    - [**Installing Formatters**](#installing-formatters)
      - [**macOS**](#macos)
      - [**Ubuntu / Debian**](#ubuntu--debian)
    - [**Running Formatters**](#running-formatters)
      - [**Option 1: Using `dev_tools.sh`**](#option-1-using-dev_toolssh)
      - [**Option 2: Running Formatters Manually**](#option-2-running-formatters-manually)
  - [**Managing `.proto` Files**](#managing-proto-files)
    - [**Generating Proto Files**](#generating-proto-files)
      - [**Option 1: Using `dev_tools.sh`**](#option-1-using-dev_toolssh-1)
      - [**Option 2: Running `protoc` Manually**](#option-2-running-protoc-manually)
  - [**Troubleshooting**](#troubleshooting)
    - [**Protobuf Runtime Version Error**](#protobuf-runtime-version-error)
      - [**Error Message**](#error-message)
      - [**Solution**](#solution)
  - [**Development Setup \& Build**](#development-setup--build)
    - [**Setup (Conda, Linux)**](#setup-conda-linux)
    - [**Building the Gradle Project (C++ and JVM)**](#building-the-gradle-project-c-and-jvm)
    - [**Building Only the JVM C++ Component**](#building-only-the-jvm-c-component)
  - [**Running Python Unit Tests with Coverage**](#running-python-unit-tests-with-coverage)
    - [**Install Dependencies**](#install-dependencies)
    - [**Run Tests with Coverage**](#run-tests-with-coverage)

---

## **Adding a New Parameter**  

To add a new parameter to **CraftGround**, follow these steps:  

1. **Edit the corresponding `.proto` file** in `proto/` directory.  
2. **Regenerate the `.proto` files** using `protoc`.  
3. **Modify Python files** in `craftground/` to support the new parameter.  

---

## **Code Formatting**  

Consistent code formatting ensures readability and maintainability.  

### **Installing Formatters**  

#### **macOS**  
```zsh
brew install ktlint clang-format google-java-format
```

#### **Ubuntu / Debian**  
```bash
wget https://apt.llvm.org/llvm.sh
sudo ./llvm.sh 19
sudo apt install clang-format-19
sudo ln -s /usr/bin/clang-format-19 /usr/bin/clang-format
```

---

### **Running Formatters**  

#### **Option 1: Using `dev_tools.sh`**  
```bash
source ./dev_tools.sh
format_code
```

#### **Option 2: Running Formatters Manually**  
```bash
git ls-files -- '*.h' '*.cpp' '*.mm' | xargs clang-format -i
git ls-files -- '*.java' -z | xargs -0 -P 4 google-java-format -i
ktlint '!**/com/kyhsgeekcode/minecraftenv/proto/**' --format
```

---

## **Managing `.proto` Files**  

Proto files define the **communication structure** between Minecraft and CraftGround.  

### **Generating Proto Files**  

#### **Option 1: Using `dev_tools.sh`**  
```bash
source ./dev_tools.sh
generate_proto
```

#### **Option 2: Running `protoc` Manually**  
```bash
cd src/
protoc proto/action_space.proto --python_out=craftground
protoc proto/initial_environment.proto --python_out=craftground
protoc proto/observation_space.proto --python_out=craftground
protoc proto/action_space.proto --java_out=craftground/MinecraftEnv/src/main/java/ --kotlin_out=craftground/MinecraftEnv/src/main/java/
protoc proto/initial_environment.proto --java_out=craftground/MinecraftEnv/src/main/java/ --kotlin_out=craftground/MinecraftEnv/src/main/java/
protoc proto/observation_space.proto --java_out=craftground/MinecraftEnv/src/main/java/ --kotlin_out=craftground/MinecraftEnv/src/main/java/
```

---

## **Troubleshooting**  

### **Protobuf Runtime Version Error**  

#### **Error Message**  
```plaintext
google.protobuf.runtime_version.VersionError: Detected incompatible Protobuf Gencode/Runtime versions when loading proto/initial_environment.proto: 
gencode 5.29.1 runtime 5.27.3. Runtime version cannot be older than the linked gencode version.
See Protobuf version guarantees at https://protobuf.dev/support/cross-version-runtime-guarantee.
```

#### **Solution**  
```bash
pip install --upgrade protobuf
```

---

## **Development Setup & Build**  

CraftGround development requires **Conda**, **Java**, **CMake**, and **C++ build tools**.  

### **Setup (Conda, Linux)**  
```bash
conda create --name craftground python=3.11
conda activate craftground
conda install gymnasium Pillow numpy protobuf typing_extensions psutil pytorch ninja build cmake
conda install -c conda-forge openjdk=21 libgl-devel
conda install glew
python -m build
```

---

### **Building the Gradle Project (C++ and JVM)**  
```bash
cd src/craftground/MinecraftEnv
./gradlew build
```

---

### **Building Only the JVM C++ Component**  
```bash
cmake src/main/cpp -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
cmake --build .
```

---

## **Running Python Unit Tests with Coverage**  

To ensure code quality, run Python unit tests with **coverage reporting**.  

### **Install Dependencies**  
```bash
python -m pip install coverage pytest
```

### **Run Tests with Coverage**  
```bash
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
