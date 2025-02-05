#include "cross_gl.h"

static float *depthPixels = nullptr;
static size_t depthPixelsSize = 0;

float *captureDepth(GLuint depthFramebufferId, int width, int height) {
    glBindFramebuffer(GL_FRAMEBUFFER, depthFramebufferId);
    const size_t newDepthPixelsSize = width * height;

    if (newDepthPixelsSize != depthPixelsSize) {
        if (depthPixels != nullptr) {
            delete[] depthPixels;
        }
        depthPixels = new float[newDepthPixelsSize];
        depthPixelsSize = newDepthPixelsSize;
    }
    glReadPixels(
        0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, depthPixels
    );
    return depthPixels;
}