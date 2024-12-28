#include <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#define GL_SILENCE_DEPRECATION
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl.h>

#include "framebuffer_capturer_apple.h"

IOSurfaceRef createSharedIOSurface(int width, int height) {
    NSDictionary *surfaceAttributes = @{
        (id)kIOSurfaceWidth : @(width),
        (id)kIOSurfaceHeight : @(height),
        (id)kIOSurfaceBytesPerElement : @(4),     // RGBA8
        (id)kIOSurfacePixelFormat : @(0x42475241) // 'RGBA'
    };

    return IOSurfaceCreate((CFDictionaryRef)surfaceAttributes);
}

static mach_port_t createMachPortForIOSurface(IOSurfaceRef ioSurface) {
    mach_port_t machPort = MACH_PORT_NULL;
    machPort = IOSurfaceCreateMachPort(ioSurface);
    return machPort;
}

static IOSurfaceRef ioSurface;
static bool initialized = false;
static GLuint textureID;

// TODO: Depth buffer
int initializeIoSurface(int width, int height, void **return_value) {
    if (initialized) {
        return 0;
    }

    // If were to use colorAttachment and depthAttachment, they
    // should be first converted to GL_TEXTURE_RECTANGLE_ARB, from GL_TEXTURE_2D
    // Therefore, use glCopyTexSubImage2D to copy the contents of the
    // framebuffer to ARB textures

    // Generate a texture
    glGenTextures(1, &textureID);
    ioSurface = createSharedIOSurface(width, height);
    mach_port_t machPort = createMachPortForIOSurface(ioSurface);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, textureID);
    CGLContextObj cglContext = CGLGetCurrentContext();
    CGLTexImageIOSurface2D(
        cglContext,
        GL_TEXTURE_RECTANGLE_ARB,
        GL_RGBA,
        width,
        height,
        GL_BGRA,
        GL_UNSIGNED_INT_8_8_8_8_REV,
        ioSurface,
        0
    );
    initialized = true;
    const int size = sizeof(machPort);
    void *bytes = malloc(size);
    if (bytes == NULL) {
        return -1;
    }
    memcpy(bytes, &machPort, size);
    *return_value = bytes;
    return size;
}

void copyFramebufferToIOSurface(int width, int height) {
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, textureID);
    glCopyTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, 0, 0, width, height);
}