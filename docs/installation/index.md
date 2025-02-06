---
title: Installation
nav_order: 2
---

# **Installation Guide**  

CraftGround requires the **latest version of CMake** to properly detect **CUDA libraries** and function correctly.  

> ⚠️ The default package manager (e.g., `apt`) **only provides an outdated version** of CMake (e.g., `3.10.2`), which is **insufficient** for CraftGround.  

To ensure a smooth setup, **install the latest CMake version** via **Conda** or **pip** instead of using system package managers.

---

## **Linux / macOS Installation**  

```bash
conda create -n exp_craftground python=3.11
conda activate exp_craftground
conda install -c conda-forge openjdk=21 cmake glew libpng zlib opengl flite
pip install craftground
```

This will:  
✅ **Create a new Conda environment** named `exp_craftground`  
✅ **Install the latest CMake** from `conda-forge`  
✅ **Install required dependencies** (`GLEW`, `OpenGL`, `libpng`, etc.)  
✅ **Install CraftGround** via `pip`  

---

## **Windows Installation**  

### **1. Enable Long File Paths (Windows Limitation Fix)**  

> 🛑 Windows has a **default path length limitation** (~260 characters).  
> You may need to enable **long file paths** to avoid issues during installation.  

#### **Option 1: Edit Registry (Manual Method)**  
1. Open `regedit` (`Windows + R`, type `regedit`, and press **Enter**).  
2. Navigate to:  
   ```
   HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
   ```
3. Find `LongPathsEnabled` and **set its value to `1`**.  
4. Restart your PC to apply changes.  

#### **Option 2: Run This Command (Automated Method)**  
Alternatively, run this command in **Command Prompt (Admin mode):**  
```cmd
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

---

### **2. Install CraftGround on Windows**  

Once long file paths are enabled, proceed with installation:  

```powershell
conda create -n exp_craftground python=3.11
conda activate exp_craftground
conda install -c conda-forge openjdk=21 cmake glew libpng zlib opengl flite
pip install craftground
```

This will:  
✅ **Create a new Conda environment**  
✅ **Install required dependencies**  
✅ **Ensure the latest version of CMake is used**  
✅ **Install CraftGround**  

---

## **Troubleshooting**  

### **CMake Not Found or Outdated Version**  
If you encounter issues where `cmake` is missing or an outdated version is detected:  

#### **Check Installed CMake Version**  
```bash
cmake --version
```
> ✅ Expected Output: `cmake version 3.21.0` or later.  

#### **Manually Install Latest CMake** (If Needed)  
```bash
pip install --upgrade cmake
```

---

## **Next Steps**  

🎯 **You're all set!**  
Now, you can proceed to:  
- [Run Your First Experiment](../configuration/index)  
- [Learn About Observation](../observation_space/index) & [Action Spaces](../action_space)