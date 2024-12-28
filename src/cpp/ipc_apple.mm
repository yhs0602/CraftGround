#include <CoreGraphics/CoreGraphics.h>
#include <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#define GL_SILENCE_DEPRECATION
#include "ipc_apple.h"
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl.h>
#include <stdexcept>

IOSurfaceRef getIOSurfaceFromMachPort(mach_port_t machPort) {
    IOSurfaceRef ioSurface = IOSurfaceLookupFromMachPort(machPort);
    return ioSurface;
}

id<MTLTexture> createMetalTextureFromIOSurface(
    id<MTLDevice> device, IOSurfaceRef ioSurface, int width, int height
) {
    MTLTextureDescriptor *descriptor = [[MTLTextureDescriptor alloc] init];
    descriptor.pixelFormat = MTLPixelFormatRGBA8Unorm;
    descriptor.width = width;
    descriptor.height = height;
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor
                                                    iosurface:ioSurface
                                                        plane:0];
    return texture;
}

static void deleteDLManagedTensor(DLManagedTensor *self) {
    free(self->dl_tensor.shape);
    free(self);
}

DLManagedTensor *
createDLPackTensor(IOSurfaceRef ioSurface, size_t width, size_t height) {
    DLManagedTensor *tensor =
        (DLManagedTensor *)malloc(sizeof(DLManagedTensor));

    tensor->dl_tensor.data = IOSurfaceGetBaseAddress(ioSurface);
    tensor->dl_tensor.ndim = 3; // H x W x C
    tensor->dl_tensor.shape = (int64_t *)malloc(3 * sizeof(int64_t));
    tensor->dl_tensor.shape[0] = height;
    tensor->dl_tensor.shape[1] = width;
    tensor->dl_tensor.shape[2] = 4; // RGBA
    tensor->dl_tensor.strides = NULL;
    tensor->dl_tensor.byte_offset = 0;

    tensor->dl_tensor.dtype =
        (DLDataType){kDLUInt, 8, 1}; // Unsigned 8-bit integer
    tensor->dl_tensor.device = (DLDevice){kDLMetal, 0}; // metal gpu

    // 메모리 해제 삭제자 설정
    tensor->deleter = deleteDLManagedTensor;
    return tensor;
}

DLManagedTensor *
mtl_tensor_from_mach_port(int machPort, int width, int height) {
    IOSurfaceRef ioSurface = getIOSurfaceFromMachPort((mach_port_t)machPort);
    if (!ioSurface) {
        throw std::runtime_error("Failed to initialize IOSurface");
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        throw std::runtime_error("Failed to create Metal device");
    }

    // id<MTLTexture> texture = createMetalTextureFromIOSurface(device,
    // ioSurface, width, height); if (!texture) {
    //     throw std::runtime_error("Failed to create Metal texture");
    // }

    DLManagedTensor *tensor = createDLPackTensor(ioSurface, width, height);

    return tensor;
}