#pragma once
#include "cross_gl.h"

void initDepthResources(int width, int height);
float *captureDepth(
    GLuint depthFramebufferId,
    int width,
    int height,
    bool requiresDepthConversion,
    float near,
    float far
);