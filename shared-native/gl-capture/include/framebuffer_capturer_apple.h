#ifndef __FRAMEBUFFER_CAPTURER_APPLE_H__

#define __FRAMEBUFFER_CAPTURER_APPLE_H__

#include <IOSurface/IOSurface.h>
#include <mach/mach.h>

// Shared with vulkan_metal_zerocopy_apple.mm, which needs its own IOSurface
// (Vulkan's VK_EXT_metal_objects import target, separate from the GL
// zerocopy IOSurface above) and its own mach port export to Python.
IOSurfaceRef createSharedIOSurface(int width, int height);
mach_port_t createMachPortForIOSurface(IOSurfaceRef ioSurface, int python_pid);

int initializeIoSurface(
    int width, int height, void **return_value, int python_pid
);
void copyFramebufferToIOSurface(
    int sourceTextureId,
    int width,
    int height,
    bool drawCursor,
    int mouseX,
    int mouseY
);

#endif