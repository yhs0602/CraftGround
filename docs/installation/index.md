# Installation
Latest cmake is required for Craftground to ensure it find the cuda libraries correctly. Currently the apt repository has cmake 3.10.2, which is not enough for Craftground. To install the latest version of cmake on linux, you should use pip or conda to install it.
```bash
conda create -n exp_craftground python=3.11
conda activate exp_craftground
conda install conda-forge::openjdk=21 conda-forge::cmake conda-forge::glew conda-forge::libpng conda-forge::libzlib conda-forge::libopengl conda-forge::libflite
pip install craftground
```


# Setup on windows
Note: you may need to enable long file path due to windows limitation. You can enable it by editing registry as mentioned [here](https://docs.python.org/3/using/windows.html#removing-the-max-path-limitation)

> In the latest versions of Windows, this limitation can be expanded to approximately 32,000 characters. Your administrator will need to activate the “Enable Win32 long paths” group policy, or set LongPathsEnabled to 1 in the registry key HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem.

The following command does.
```cmd
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
```


```ps
conda create -n exp_craftground python=3.11
conda activate exp_craftground
conda install conda-forge::openjdk=21 conda-forge::cmake conda-forge::glew conda-forge::libpng conda-forge::libzlib conda-forge::libopengl conda-forge::libflite
pip install craftground
```
