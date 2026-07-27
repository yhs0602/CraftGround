from enum import Enum


class ScreenEncodingMode(Enum):
    RAW = 0
    PNG = 1
    # Both ZEROCOPY_* variants share the same native mach-port/CUDA-handle capture
    # path (see FramebufferCapturer.kt) and only differ in which array type the
    # shared GPU buffer gets wrapped as on the Python side.
    ZEROCOPY_TORCH = 2
    ZEROCOPY_JAX = 3
