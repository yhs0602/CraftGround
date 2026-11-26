#pragma once
#include <jni.h>
#include "cross_gl.h"

// CPU-only (no GL calls) tail of the capture pipeline, shared by the OpenGL
// path (framebuffer_capturer.cpp, fed by glReadPixels) and the backend-neutral
// Blaze3D path (convertCapturedFrameImpl below, fed by
// CommandEncoder.copyTextureToBuffer). See docs/26_2_vulkan_capture.md.
//
// Nearest-neighbour downscale of a tightly packed RGB8 image. Returns a newly
// allocated buffer the caller owns.
extern "C" GLubyte *resize_pixels(
    jint &textureWidth,
    jint &textureHeight,
    jint &targetSizeX,
    jint &targetSizeY,
    GLubyte *pixels
);

// Resize (if needed) -> draw cursor -> encode as RAW/PNG -> wrap in a
// protobuf ByteString. `pixels` is tightly packed RGB8 of srcWidth x srcHeight
// and stays owned by the caller; any intermediate buffer allocated here is
// freed here. Returns nullptr (with a pending exception, or none) on failure.
jobject makeByteStringFromRgb(
    JNIEnv *env,
    GLubyte *pixels,
    jint srcWidth,
    jint srcHeight,
    jint targetSizeX,
    jint targetSizeY,
    jint encodingMode,
    jboolean drawCursor,
    jint xPos,
    jint yPos
);
