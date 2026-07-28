#include "cross_gl.h"
#include "depth_capture.h"

// mc262 fork of shared-native/gl-capture/depth_capture.cpp
// (docs/26_2_phase2_plan.md W3, depth path). Three things changed versus mc121:
//
// 1. Texture-based instead of FBO-based. 26.2 replaced Framebuffer with
//    GpuTexture/RenderTarget, so there is no framebuffer integer to hand over -
//    only a GL texture id from GlTexture.glId(). Like rgb_capture.cpp, this file
//    owns a single capture FBO and attaches whatever depth texture id it is
//    given as GL_DEPTH_ATTACHMENT, re-attaching only when the id changes.
//
// 2. Reverse-Z. Minecraft 26.2 renders the level with a reversed depth range:
//    Projection.getMatrix() feeds zFar as JOML's `near` and zNear as its `far`,
//    GameRenderer clears the depth texture to 0.0, and the depth test is
//    GL_GREATER. So a raw depth of 1.0 is the NEAR plane and 0.0 is the FAR
//    plane - the opposite of mc121. The linearization below is derived for that
//    convention (and for both clip ranges; see zZeroToOne).
//
// 3. CPU-only linearization. mc121 had a GPU path that ran a full-screen quad
//    through its own shader program. That is not safe to do here: this capture
//    runs in the middle of GameRenderer.render (it has to - see the mixin), and
//    26.2's GlStateManager caches program/VAO/viewport state, so issuing raw
//    draw calls mid-frame would desync that cache and corrupt the GUI pass that
//    follows. Only the framebuffer binding is touched, and it is restored. The
//    readback is a pipeline stall either way, so doing the arithmetic on the CPU
//    costs little on top.

static float *depthPixels = nullptr;
static size_t depthPixelsSize = 0;
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
            GL_DEPTH_ATTACHMENT,
            GL_TEXTURE_2D,
            textureId,
            0
        );
        // Depth-only FBO: without this some drivers reject the read as coming
        // from a missing color attachment.
        glReadBuffer(GL_NONE);
        attachedTextureId = textureId;
    }
    return captureFbo;
}

// Raw window-space depth -> linear eye-space distance, normalized to [0, 1] by
// the far plane, matching mc121's output convention (near -> ~0, far -> 1).
//
// 26.2 builds the projection with JOML's near/far swapped (reverse-Z), so with
// N = real zNear and F = real zFar:
//
//   clip range [0, 1] (GL_ARB_clip_control, the normal desktop case):
//     z = N*F / (N + d*(F - N))
//   clip range [-1, 1] (no clip control):
//     z = 2*N*F / (N + F + (2d - 1)*(F - N))
//
// Both give z = F at d = 0 (the cleared/far value) and z = N at d = 1.
static float linearizeReverseZ(float d, float n, float f, bool zZeroToOne) {
    if (zZeroToOne) {
        return (n * f) / (n + d * (f - n));
    }
    return (2.0f * n * f) / (n + f + (2.0f * d - 1.0f) * (f - n));
}

float *captureDepth(
    GLuint depthTextureId,
    int width,
    int height,
    bool requiresDepthConversion,
    float nearPlane,
    float farPlane,
    bool zZeroToOne
) {
    GLint previousReadFbo = 0;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &previousReadFbo);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, ensureCaptureFbo(depthTextureId));

    const size_t newDepthPixelsSize = (size_t)width * (size_t)height;
    if (newDepthPixelsSize != depthPixelsSize) {
        if (depthPixels != nullptr) {
            delete[] depthPixels;
        }
        depthPixels = new float[newDepthPixelsSize];
        depthPixelsSize = newDepthPixelsSize;
    }

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(
        0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, depthPixels
    );

    if (requiresDepthConversion) {
        for (size_t i = 0; i < newDepthPixelsSize; i++) {
            depthPixels[i] =
                linearizeReverseZ(
                    depthPixels[i], nearPlane, farPlane, zZeroToOne
                ) /
                farPlane;
        }
    }

    glBindFramebuffer(GL_READ_FRAMEBUFFER, (GLuint)previousReadFbo);
    return depthPixels;
}
