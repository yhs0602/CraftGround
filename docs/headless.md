# How to run minecraft environments on headless server

To run Minecraft environments on a headless server with GPU acceleration, you need to emulate a 3D X server (and of course, 2D X server.) Here, we use [VirtualGL](https://virtualgl.org/) to emulate the 3D X server.

## Installing VirtualGL
VirtualGL is incompatible with Wayland, so you need to check before installing it. If you are using a Wayland session, you need to switch to an X11 session.
### Check if you are using wayland session
```shell
echo $WAYLAND_DISPLAY
echo $XDG_SESSION_TYPE
ps aux | grep -E 'weston|sway'
```
If nothing is printed, proceed with the below steps.
### Installing VirtualGL
```shell
wget https://sourceforge.net/projects/virtualgl/files/3.1/virtualgl_3.1_amd64.deb/download
mv download vgl3.1.deb
sudo dpkg -i vgl3.1.deb
```

### Configuring VirtualGL
```shell
sudo vglserver_config
```
You can choose the first option (install both GLX and EGL), and then choose "do not restrict access to only vglusers" for the rest of the configuration.
You may meet the message such as 
> modprobe: FATAL: Module nvidia_drm is in use. You must execute 'modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia' with the display manager stopped in order for the new device permission settings to become effective.

Then execute the below command
```shell
# Stop the display manager
sudo systemctl stop gdm
# Unload the nvidia modules
sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia
```
If you meet the message such as
> modprobe: FATAL: Module nvidia_drm is in use.

Then execute the below command
```shell
sudo lsof /dev/nvidia*
```
To get the pid of the offending process. Then kill the process.
```shell
pkill <pid> && sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia
```
Now restart the display manager.
```shell
# Load the nvidia modules again
sudo modprobe nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo systemctl restart gdm
```

### Installing Xvfb
Xvfb is required to emulate a 2D X server.
```shell
sudo apt install xvfb
```

### Starting Xvfb buffer
Use the below command to start the Xvfb buffer.
```shell
Xvfb :2 -screen 0 1024x768x24 +extension GLX -ac +extension RENDER &
export DISPLAY=:2
```

### Validating VirtualGL acceleration
You can run this utility to check if the GPU is used.
```shell
VGL_DISPLAY=:0 vglrun /opt/VirtualGL/bin/glxspheres64
```
Here you may see that it is still using software rendering, such as `llvmpipe`. Then follow the below steps.

### Configuring the GPU for VirtualGL
First check the BusID of the GPU using this command:
```shell
sudo nvidia-xconfig --query-gpu-info
```
Let's say it is `PCI:104:0:0`. Now use this command to configure the GPU for VirtualGL. Replace `PCI:104:0:0` with your BusID.
```shell
sudo nvidia-xconfig -a --allow-empty-initial-configuration --use-display-device=None --virtual=1920x1200 --busid PCI:104:0:0
```
Now restart the display manager.
```shell
sudo systemctl restart gdm
```

### Check if the GPU is used again
```shell
vglrun /opt/VirtualGL/bin/glxspheres64
```
Now it will say something like `OpenGL Renderer: NVIDIA GeForce RTX 3090/PCIe/SSE2`.

## Prevent FPS drop to 1
You may have to disable PDMS and screen saver to prevent FPS drop to 1. First, backup your current settings.
```shell
sudo cp /etc/X11/xorg.conf /etc/X11/xorg.conf.bak
```
Then comment out all the DPMS settings in the xorg.conf file.
```text
Section "Monitor"
    Identifier     "Monitor0"
    VendorName     "Unknown"
    ModelName      "Unknown"
    HorizSync       28.0 - 33.0
    VertRefresh     43.0 - 72.0
#   Option         "DPMS"
EndSection
```
Restart the display manager.
```shell
sudo systemctl restart gdm
```
Now disable the screen saver and DPMS features from the display manager.
```shell
export DISPLAY=:0.0
xset -q # Check if the screen saver is disabled
xset s off
xset -dpms # Disable DPMS (Energy Star) features.
xset -q # Check if the screen saver is disabled
export DISPLAY=:2 # Change back to the display to the 2D X server
```
This effectively disables the screen saver and DPMS features. You may want to add these commands to your startup scripts such as `.xinitrc` or `.xprofile` to make it permanent even after reboot.