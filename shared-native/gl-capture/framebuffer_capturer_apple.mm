#include <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#define GL_SILENCE_DEPRECATION
#include <OpenGL/OpenGL.h> // CGLGetCurrentContext / CGLTexImageIOSurface2D
#include <servers/bootstrap.h>

// cross_gl.h (rather than the legacy <OpenGL/gl.h>) for glBindFramebuffer /
// glFramebufferTexture2D / glBlitFramebuffer - see the comment on targetFbo
// below for why this needs GL 3.0 core FBO entry points.
#include "cross_gl.h"
#include "cursor.h"
#include "framebuffer_capturer_apple.h"
#include "iosurface_message.h"

IOSurfaceRef createSharedIOSurface(int width, int height) {
    NSDictionary *surfaceAttributes = @{
        (id)kIOSurfaceWidth : @(width),
        (id)kIOSurfaceHeight : @(height),
        (id)kIOSurfaceBytesPerElement : @(4),     // RGBA8
        (id)kIOSurfacePixelFormat : @(0x52474241) // 'RGBA'
    };

    return IOSurfaceCreate((CFDictionaryRef)surfaceAttributes);
}

mach_port_t createMachPortForIOSurface(IOSurfaceRef ioSurface, int python_pid) {
    mach_port_t surface = IOSurfaceCreateMachPort(ioSurface);
    if (!MACH_PORT_VALID(surface))
        return MACH_PORT_NULL;
    task_t pythonTask = MACH_PORT_NULL;
    kern_return_t result =
        task_for_pid(mach_task_self(), python_pid, &pythonTask);
    mach_port_name_t name = surface;
    if (result == KERN_SUCCESS) {
        if (const char *receiver = getenv("CRAFTGROUND_IOSURFACE_RECEIVER")) {
            name =
                static_cast<mach_port_name_t>(strtoul(receiver, nullptr, 10));
            mach_port_t destination;
            mach_msg_type_name_t type;
            result = mach_port_extract_right(
                pythonTask, name, MACH_MSG_TYPE_MAKE_SEND, &destination, &type
            );
            if (result == KERN_SUCCESS) {
                IOSurfaceMessage message{};
                message.header.msgh_bits =
                    MACH_MSGH_BITS_COMPLEX |
                    MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, 0);
                message.header.msgh_size = sizeof(message);
                message.header.msgh_remote_port = destination;
                message.header.msgh_id = IOSurfaceMessageId;
                message.body.msgh_descriptor_count = 1;
                message.surface.name = surface;
                message.surface.disposition = MACH_MSG_TYPE_COPY_SEND;
                message.surface.type = MACH_MSG_PORT_DESCRIPTOR;
                result = mach_msg(
                    &message.header,
                    MACH_SEND_MSG | MACH_SEND_TIMEOUT,
                    sizeof(message),
                    0,
                    MACH_PORT_NULL,
                    0,
                    MACH_PORT_NULL
                );
                mach_port_deallocate(mach_task_self(), destination);
            }
        } else {
            // Older Python packages expect a send right at the published name.
            result = mach_port_insert_right(
                pythonTask, name, surface, MACH_MSG_TYPE_COPY_SEND
            );
        }
        mach_port_deallocate(mach_task_self(), pythonTask);
    }
    mach_port_deallocate(mach_task_self(), surface);
    if (result != KERN_SUCCESS) {
        fprintf(
            stderr,
            "Failed to share IOSurface with Python: %s\n",
            mach_error_string(result)
        );
        return MACH_PORT_NULL;
    }
    return name;
}

static IOSurfaceRef ioSurface;
static bool initialized = false;
static GLuint textureID;
// Destination FBO wrapping textureID (used both for the blit target and for
// the cursor overlay - see below) and a source FBO the caller's texture gets
// attached to each frame. Blit rather than glCopyImageSubData: macOS's
// OpenGL.framework tops out around a GL 4.1 core profile, and
// glCopyImageSubData is only core as of GL 4.3 (GL_ARB_copy_image support is
// not guaranteed there), whereas glBlitFramebuffer has been core since GL 3.0
// and blits between differing texture targets (GL_TEXTURE_2D source,
// GL_TEXTURE_RECTANGLE dest here) without needing them to match.
static GLuint targetFbo;
static GLuint sourceFbo;

// On 26.2 there is no long-lived FBO id available from Java for the main
// color target (the RAW/PNG path is texture-based, see
// docs/26_2_vulkan_capture.md), so the synthetic agent cursor is composited
// onto this copy of the frame after the blit below, rather than onto the live
// framebuffer as mc121 does. This only affects what the agent sees via
// IOSurface, not what's shown in the game window.

// TODO: Depth buffer
int initializeIoSurface(
    int width, int height, void **return_value, int python_pid
) {
    if (initialized) {
        return 0;
    }

    ioSurface = createSharedIOSurface(width, height);
    if (!ioSurface)
        return -1;
    mach_port_t machPort = createMachPortForIOSurface(ioSurface, python_pid);
    if (!MACH_PORT_VALID(machPort)) {
        CFRelease(ioSurface);
        ioSurface = nullptr;
        return -1;
    }
    glGenTextures(1, &textureID);
    printf("\n\nmachPort: %u\n\n\n", machPort);
    fflush(stdout);
    glBindTexture(GL_TEXTURE_RECTANGLE, textureID);
    CGLContextObj cglContext = CGLGetCurrentContext();
    CGLTexImageIOSurface2D(
        cglContext,
        GL_TEXTURE_RECTANGLE,
        GL_RGBA,
        width,
        height,
        GL_BGRA,
        GL_UNSIGNED_INT_8_8_8_8_REV,
        ioSurface,
        0
    );

    glGenFramebuffers(1, &targetFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, targetFbo);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, textureID, 0
    );
    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Re-attached to the caller's texture every frame in
    // copyFramebufferToIOSurface (that texture id can change across resizes),
    // so it's left incomplete/empty here.
    glGenFramebuffers(1, &sourceFbo);

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

void copyFramebufferToIOSurface(
    int sourceTextureId,
    int width,
    int height,
    bool drawCursor,
    int mouseX,
    int mouseY
) {
    // mc262's main color target is a plain GL_TEXTURE_2D (see
    // EnvironmentInitializer's Blaze3D backend detection); blit it into the
    // IOSurface-backed GL_TEXTURE_RECTANGLE texture without ever needing
    // an FBO id from the caller.
    glBindFramebuffer(GL_READ_FRAMEBUFFER, sourceFbo);
    glFramebufferTexture2D(
        GL_READ_FRAMEBUFFER,
        GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D,
        sourceTextureId,
        0
    );
    assert(
        glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE
    );
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, targetFbo);
    glBlitFramebuffer(
        0,
        0,
        width,
        height,
        0,
        0,
        width,
        height,
        GL_COLOR_BUFFER_BIT,
        GL_NEAREST
    );
    assert(glGetError() == GL_NO_ERROR);

    if (drawCursor) {
        // Still bound as GL_DRAW_FRAMEBUFFER from the blit above.
        glViewport(0, 0, width, height);
        renderCursor(mouseX, mouseY);
    }

    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}