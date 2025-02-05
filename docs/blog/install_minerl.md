---
title: Complete Guide to Installing MineRL
parent: Blog
---

# Brief Introduction
## MineRL 0.4.4

This may help solving issues such as https://github.com/minerllabs/minerl/issues/788.
- Use this table to get latest gcc version your OS supports: https://askubuntu.com/a/1163021/901082
- To solve MixinGradle issue, follow the steps as mentioned here: https://github.com/MineDojo/MineDojo/issues/113#issuecomment-1908997704
```bash
conda create -n exp_minerl044 python=3.11
conda activate exp_minerl044
conda install conda-forge::openjdk=8 
pip install setuptools==65.5.0 pip==21 wheel==0.38.0
pip install gym==0.19.0
pip install --upgrade pip
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install -y libxml2-dev libxslt1-dev gfortran libopenblas-dev software-properties-common
# Ensure you have the latest version of gcc:
# To check the version of gcc, run `gcc --version`
sudo apt-get install gcc-10 g++-10
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 120
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 120
OPENSSL_ROOT_DIR=$CONDA_PREFIX pip download --no-binary :all: minerl==0.4.4
tar -xvf minerl-0.4.4.tar.gz
cd minerl-0.4.4
# remove gym line from requirements.txt
pip install -r requirements.txt
# Edit minerl-0.4.4/minerl/Malmo/Minecraft/build.gradle:L19 based on
# https://github.com/MineDojo/MineDojo/issues/113#issuecomment-1908997704
# classpath('MixinGradle-dcfaf61:MixinGradle:dcfaf61'){ // 0.6
# Add repository maven to the build.gradle
#         maven { url 'file:file:/absolute-path/to/that/repo's/parent' }
pip install .
pip install wandb tensorboard moviepy stable-baselines3
pip install --upgrade git+https://github.com/DLR-RM/stable-baselines3.git # To ensure correct video rendering
vglrun python experiments/minerl044.py --image_width 64x64 --load simulation
```


## MineRL 1.0.0

Installing MineRL 1.0.0 is much easier than 0.4.4.
```bash
conda create -n exp_minerl100 python=3.11
conda activate exp_minerl100
conda install conda-forge::openjdk=8
pip install git+https://github.com/minerllabs/minerl
pip install wandb tensorboard moviepy stable-baselines3
pip install --upgrade git+https://github.com/DLR-RM/stable-baselines3.git # To ensure correct video rendering
conda install -c anaconda cudnn # for ppo
# On cuda devices
pip install jax[cuda]
# On apple devices
pip install jax-metal
vglrun python experiments/minerl100_exp.py --image_width 64x64 --load simulation
```

## MineRL 1.0.0 on MacOS
We should apply  https://github.com/MineDojo/MineDojo/pull/56/ to make it work on MacOS.
```gradle
    def schemaIndexFile = new File('src/main/resources/schemas.index')
```
to
```gradle
    def schemaIndexFile = new File("$projectDir/src/main/resources/schemas.index")
```
or use the patched version:
```
pip install git+https://github.com/yhs0602/minerl  
```

# Troubleshooting
## Installing MineRL 0.4.4
### Error because of MixinGradle
Edit minerl-0.4.4/minerl/Malmo/Minecraft/build.gradle:L19 based on https://github.com/MineDojo/MineDojo/issues/113#issuecomment-1908997704
```gradle
classpath('MixinGradle-dcfaf61:MixinGradle:dcfaf61'){ // 0.6
```
Add repository maven to the build.gradle
```
maven { url 'file:file:/absolute-path/to/that/repo's/parent' }
```

### Hangs in installing build dependencies
```bash
pip install --upgrade pip
```

### OpenSSL error
Append `OPENSSL_ROOT_DIR=$CONDA_PREFIX` to the command if you are using conda. Otherwise, you can set the environment variable after installing openssl-dev.
```bash
OPENSSL_ROOT_DIR=$CONDA_PREFIX pip download --no-binary :all: minerl==0.4.4
```

### NumPy requires GCC >= 8.4, SciPy requires GCC >= 9.1
- Use this table to get latest gcc version your OS supports: https://askubuntu.com/a/1163021/901082

```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install -y software-properties-common
sudo apt-get install gcc-10 g++-10
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 120
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 120
```

### gfortran not found
```bash
sudo apt install gfortran
```

### OpenBLAS not found
```bash
sudo apt install libopenblas-dev
```


## Installing MineRL 1.0.0
Ensure you have the correct version of JDK. MineRL 1.0.0 requires JDK 8.
```bash
conda install conda-forge::openjdk=8
```
or 
```bash
sudo apt install openjdk-8-jdk
```
or 
```bash
jenv local 1.8
```
You can get the JDK 8 from various sources such as:
- https://www.azul.com/downloads/?version=java-8-lts&package=jdk#zulu

## Installing Craftground
Ensure you have the latest version of cmake. Currently the apt repository has cmake 3.10.2, which is not enough for Craftground. To install the latest version of cmake, you should use pip or conda to install it.
```bash
conda install cmake
```
or 
```bash
pip install --upgrade cmake
```


## Malmö, MineRL, and CraftGround is not using GPU on CUDA devices
You should install VirtualGL and run the experiments. Take a look at this MineRL documentation:

- https://minerl.readthedocs.io/en/latest/notes/performance-tips.html#faster-alternative-to-xvfb

Also you can check this guide:

- https://yhs0602.github.io/CraftGround/headless.html

```bash
echo $WAYLAND_DISPLAY
echo $XDG_SESSION_TYPE
ps aux | grep -E ’weston|sway’
sudo apt install virtualgl
wget https://sourceforge.net/projects/virtualgl/files/3.1/virtualgl_3.1_amd64.
deb/download
mv download vgl3.1.deb
sudo dpkg -i vgl3.1.deb
sudo vglserver_config
# During configuration, select the option to install both GLX and EGL and adjust device permissions as required
# In case you meed the following error, run the following command
# modprobe: FATAL: Module nvidia_drm is in use. You must execute modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia’ with the display manager stopped in order for the new device permission settings to become effective.
sudo systemctl stop gdm
sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia
# If you meet modprobe: FATAL: Module nvidia_drm is in use.
sudo lsof /dev/nvidia*
pkill <pid>
# Restart the display manager
sudo modprobe nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo systemctl restart gdm
# Install Xvfb
sudo apt install xvfb
Xvfb :2 -screen 0 1024x768x24 +extension GLX -ac +extension RENDER & 
export DISPLAY=:2
VGL_DISPLAY=:0 vglrun /opt/VirtualGL/bin/glxspheres64
sudo nvidia-xconfig --query-gpu-info
sudo nvidia-xconfig -a --allow-empty-initial-configuration \
--use-display-device=None --virtual=1920x1200 \
--busid PCI:<BusID>
sudo systemctl restart gdm
VGL_DISPLAY=:0 vglrun /opt/VirtualGL/bin/glxspheres64
# OpenGL Renderer: NVIDIA GeForce RTX 3090/PCIe/SSE2.
```
