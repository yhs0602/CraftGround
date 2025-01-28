# CraftGround - Reinforcement Learning Environment for Minecraft
[![Wheels (All))](https://github.com/yhs0602/CraftGround/actions/workflows/publish-upload.yml/badge.svg)](https://github.com/yhs0602/CraftGround/actions/workflows/publish-upload.yml)
[![Python package](https://github.com/yhs0602/CraftGround/actions/workflows/python-ci.yml/badge.svg)](https://github.com/yhs0602/CraftGround/actions/workflows/python-ci.yml)
[![CMake Build](https://github.com/yhs0602/CraftGround/actions/workflows/cmake-build.yml/badge.svg)](https://github.com/yhs0602/CraftGround/actions/workflows/cmake-build.yml)
[![Gradle Build](https://github.com/yhs0602/CraftGround/actions/workflows/gradle.yml/badge.svg)](https://github.com/yhs0602/CraftGround/actions/workflows/gradle.yml)
[![Coverage Badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/yhs0602/8497c0c395a8d6b18d1e81f05ff57dba/raw/craftground__heads_main.json)]()

<img src="docs/craftground.webp" alt="CraftGround_Logo" width="50%"/>


[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fyhs0602%2FMinecraftRL)](https://github.com/yhs0602/MinecraftRL)

**CraftGround** provides a lightweight and customizable environment for reinforcement learning experiments using Minecraft.

---

- [Installation](#installation)
  - [Quick start (conda / mamba)](#quick-start-conda--mamba)
  - [Quick start (Ubuntu based systems)](#quick-start-ubuntu-based-systems)
  - [Install development version](#install-development-version)
- [Run your first experiment](#run-your-first-experiment)
  - [Example repositories](#example-repositories)
  - [Example code](#example-code)
- [Environment Specifications](#environment-specifications)
  - [How to execute minecraft command in a gymnasium wrapper?](#how-to-execute-minecraft-command-in-a-gymnasium-wrapper)
- [Technical Details](#technical-details)
- [License and Acknowledgements](#license-and-acknowledgements)
  - [Devaju font license](#devaju-font-license)
  - [Gamma Utils](#gamma-utils)
  - [Fabric-Carpet](#fabric-carpet)
- [Development / Customization](#development--customization)

---

## Installation
### Quick start (conda / mamba)
```bash
conda create -n my_experiment_env python=3.11
conda activate my_experiment_env
conda install conda-forge::openjdk=21 conda-forge::cmake conda-forge::glew-dev conda-forge::libpng conda-forge::libzlib conda-forge::libopengl conda-forge::libflite
pip install craftground
```

### Quick start (Ubuntu based systems)
Refer to the provided [Dockerfile](./Dockerfile) for a complete setup.

```bash
sudo apt-get update
sudo apt-get install -y openjdk-21-jdk python3-pip git \
    libgl1-mesa-dev libegl1-mesa-dev libglew-dev \ 
    libglu1-mesa-dev xorg-dev libglfw3-dev xvfb
apt-get clean
pip3 install --upgrade pip
pip3 install cmake # You need latest cmake, not the one provided by apt-get
pip3 install craftground
```

### Windows Setup
#### Resolve Windows Path Length Limitation
Note: you may need to enable long file path due to windows limitation. You can enable it by editing registry as mentioned [here](https://docs.python.org/3/using/windows.html#removing-the-max-path-limitation)

> In the latest versions of Windows, this limitation can be expanded to approximately 32,000 characters. Your administrator will need to activate the “Enable Win32 long paths” group policy, or set LongPathsEnabled to 1 in the registry key HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem.

The following command does the same as the above instruction:
```cmd
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

#### Resolve Conda Installation Issues
Using conda is recommended for Windows users. You can install conda from [here](https://docs.anaconda.com/miniconda/). Make sure to execute
```powershell
powershell -ExecutionPolicy Bypass
conda init powershell
```
if you are using PowerShell, in the anaconda powershell prompt.


### Setup Headless Environment (Linux)
Refer to [Headless Environment Setup](docs/headless.md) for setting up a headless environment.

### Install development version
```bash
pip install git+https://github.com/yhs0602/CraftGround.git@dev
```

## Run your first experiment
### Example repositories
- Check [the example repository](https://github.com/yhs0602/minecraft-simulator-benchmark) for benchmarking experiments.

### Example code
```python
from craftground import craftground
from stable_baselines3 import A2C

# Initialize environment
env = craftground.make(port=8023, isWorldFlat=True, ...)

# Train model
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("a2c_craftground")
```

## Environment Specifications
For detailed specifications, refer to the following documents:

- [Initial Environment](docs/initial_environment.md)
- [Observation Space](docs/observation_space.md)
- [Action Space](docs/action_space.md)


## Technical Details
See [Technical Details](docs/technical_details.md) for detailed technical information.


## License and Acknowledgements
This project is licensed under the LGPL v3.0 license. The project includes code from the following sources:

### Devaju fonts
- Source: [Dejavu Fonts](https://dejavu-fonts.github.io/License.html)

### Gamma Utils

This project includes code licensed under the GNU Lesser General Public License v3.0:
- Source: [Gamma Utils project](https://github.com/Sjouwer/gamma-utils)
<!-- - `com.kyhsgeekcode.minecraftenv.mixin.GammaMixin` originates from the  -->

### Fabric-Carpet
This project includes code from the Fabric Carpet project, licensed under the MIT License:
- Source: [Fabric-Carpet Project](https://github.com/gnembon/fabric-carpet)


## Development / Customization
For detailed development and customization instructions, see [Develop](docs/develop.md).