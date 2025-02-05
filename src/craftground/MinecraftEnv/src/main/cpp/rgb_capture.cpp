#include "cross_gl.h"

static GLubyte *rgbPixels = nullptr;
static size_t rgbPixelsSize = 0;

// **Note**: Flipping should be done in python side.
GLubyte *caputreRGB(int frameBufferId, int textureWidth, int textureHeight) {
    glBindFramebuffer(GL_READ_FRAMEBUFFER, frameBufferId);
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