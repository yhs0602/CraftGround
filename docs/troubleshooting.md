---
title: Troubleshooting  
nav_order: 6  
---

# **Troubleshooting Guide**  

This guide provides solutions to common issues encountered while using **CraftGround**.  

---

## **1. FileExistsError: Socket File Already Exists**  

### **Error Message**  
```plaintext
FileExistsError: Socket file /tmp/minecraftrl_8001.sock already exists. Please choose another port.
```

### **Cause**  
This error occurs when a **previous CraftGround session** was not properly terminated, leaving behind a **stale socket file**.  

### **Solution**  
1. **Manually remove the stale socket file:**  
   ```bash
   rm /tmp/minecraftrl_8001.sock
   ```
2. **Restart CraftGround** and check if the issue persists.

---

## **2. Zombie Minecraft Process Still Running**  

### **Issue**  
CraftGround may fail to start properly if a **Minecraft process** is still running in the background, preventing a new instance from launching.  

### **Solution**  

1. **Find the Minecraft process:**  
   ```bash
   jps -l
   ```
   > Look for a process with a name like:  
   > **`DevLaunchInjector.Main`**  

2. **Terminate the process:**  
   ```bash
   kill -9 <pid>
   ```
   Replace `<pid>` with the **process ID** from the previous step.  

3. **Restart CraftGround.**

---

### **Need More Help?**  
If you encounter additional issues, please open an issue on the **[GitHub repository](https://github.com/yhs0602/CraftGround/issues)**. 🚀  
