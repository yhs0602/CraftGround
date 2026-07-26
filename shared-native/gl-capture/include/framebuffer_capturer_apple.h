#ifndef __FRAMEBUFFER_CAPTURER_APPLE_H__

#define __FRAMEBUFFER_CAPTURER_APPLE_H__

int initializeIoSurface(
    int width,
    int height,
    void **return_value,
    int python_pid
); // , int colorAttachment, int depthAttachment
void copyFramebufferToIOSurface(int width, int height);

#endif