#include "cross_gl.h"

// mc262 fork of shared-native/gl-capture/rgb_capture.cpp
// (docs/26_2_phase2_plan.md W3). 26.2 replaced the old Framebuffer/raw-FBO-int
// model with GpuTexture/RenderTarget, so there is no FBO integer to hand over
// from the Java side anymore. Instead this file owns a single capture FBO and
// attaches whatever color texture id it's given as GL_COLOR_ATTACHMENT0, only
// re-attaching when the texture id actually changes (e.g. a window resize
// creates a new backing texture in Minecraft.renderFrame's RenderTarget). The
// exported signature (int, int, int) is unchanged from mc121's so rgb_capture.h
// didn't need to change - only the meaning of the first argument did (texture
// id instead of framebuffer id).
static GLubyte *rgbPixels = nullptr;
static size_t rgbPixelsSize = 0;
static GLuint captureFbo = 0;
static GLuint attachedTextureId = 0;

static GLuint ensureCaptureFbo(GLuint textureId) {
    if (captureFbo == 0) {
        glGenFramebuffers(1, &captureFbo);
    }
    if (attachedTextureId != textureId) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, captureFbo);
        glFramebufferTexture2D(
            GL_READ_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D,
            textureId,
            0
        );
        attachedTextureId = textureId;
    }
    return captureFbo;
}

// **Note**: Flipping should be done in python side.
GLubyte *caputreRGB(int textureId, int textureWidth, int textureHeight) {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, ensureCaptureFbo((GLuint)textureId));
    const size_t newRgbPixelsSize = textureWidth * textureHeight * 3;
    if (newRgbPixelsSize != rgbPixelsSize) {
        if (rgbPixels != nullptr) {
            delete[] rgbPixels;
        }
        rgbPixels = new GLubyte[newRgbPixelsSize];
        rgbPixelsSize = newRgbPixelsSize;
    }
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(
        0, 0, textureWidth, textureHeight, GL_RGB, GL_UNSIGNED_BYTE, rgbPixels
    );
    return rgbPixels;
}
