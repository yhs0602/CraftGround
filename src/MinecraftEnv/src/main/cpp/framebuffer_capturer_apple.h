#ifndef __FRAMEBUFFER_CAPTURER_APPLE_H__

#define __FRAMEBUFFER_CAPTURER_APPLE_H__

int initializeIoSurface(int width, int height);
void captureFramebuffer(int width, int height);

#endif