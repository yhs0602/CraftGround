#include <CoreGraphics/CoreGraphics.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#define GL_SILENCE_DEPRECATION
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl.h>

#include "framebuffer_capturer_apple.h"


IOSurfaceRef createSharedIOSurface(int width, int height) {
    NSDictionary *surfaceAttributes = @{
        (id)kIOSurfaceWidth: @(width),
        (id)kIOSurfaceHeight: @(height),
        (id)kIOSurfaceBytesPerElement: @(4),  // RGBA8
        (id)kIOSurfacePixelFormat: @(0x42475241) // 'RGBA'
    };

    return IOSurfaceCreate((CFDictionaryRef)surfaceAttributes);
}

static mach_port_t createMachPortForIOSurface(IOSurfaceRef ioSurface) {
    mach_port_t machPort = MACH_PORT_NULL;
    machPort = IOSurfaceCreateMachPort(ioSurface);
    return machPort;
}

static IOSurfaceRef ioSurface;
static GLuint textureID;
static bool initialized = false;    

int initializeIoSurface(int width, int height) {
    if (initialized) {
        return 0;
    }

    ioSurface = createSharedIOSurface(width, height);
    glGenTextures(1, &textureID);
    mach_port_t machPort = createMachPortForIOSurface(ioSurface);
    initialized = true;
    return machPort;
}

void captureFramebuffer(int width, int height) {
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB, textureID);
    CGLContextObj cglContext = CGLGetCurrentContext();
    CGLTexImageIOSurface2D(cglContext, GL_TEXTURE_RECTANGLE_ARB, GL_RGBA,
                           width,
                           height,
                           GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV,
                           ioSurface, 0);
}