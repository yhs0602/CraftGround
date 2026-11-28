#pragma once
#include "cross_gl.h"

// mc262-local replacement for shared-native/gl-capture/include/depth_capture.h.
// See depth_capture.cpp in this directory for why the signature changed.
float *captureDepth(
    GLuint depthTextureId,
    int width,
    int height,
    bool requiresDepthConversion,
    float nearPlane,
    float farPlane,
    bool zZeroToOne
);
