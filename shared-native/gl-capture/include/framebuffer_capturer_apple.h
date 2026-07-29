#ifndef __FRAMEBUFFER_CAPTURER_APPLE_H__

#define __FRAMEBUFFER_CAPTURER_APPLE_H__

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