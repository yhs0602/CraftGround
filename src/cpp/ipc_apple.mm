#include <CoreGraphics/CoreGraphics.h>
#include <IOSurface/IOSurface.h>
#import <Metal/Metal.h>
#define GL_SILENCE_DEPRECATION
#include "ipc_apple.h"
#include <OpenCL/cl_ext.h>
#include <OpenCL/opencl.h>
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Sad news: PyTorch does not support from_dlpack for Metal tensors.
// Therefore, we should create a OpenCL dlpack tensor from the IOSurface.
#define USE_OPENCL_DL_PACK_TENSOR 0

#define USE_CUSTOM_DL_PACK_TENSOR 1

IOSurfaceRef getIOSurfaceFromMachPort(mach_port_t machPort) {
    mach_port_type_t portType;
    kern_return_t result =
        mach_port_type(mach_task_self(), machPort, &portType);
    if (result != KERN_SUCCESS) {
        std::string error_msg = "Failed to query Mach Port type: ";
        error_msg += mach_error_string(result);
        throw std::runtime_error(error_msg);
    }
    if (!(portType & MACH_PORT_TYPE_SEND)) {
        std::string error_msg = "Mach Port does not have SEND rights. Type: 0x";
        error_msg += std::to_string(portType);
        throw std::runtime_error(error_msg);
    }

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

#if USE_OPENCL_DL_PACK_TENSOR
static bool initializedOpenCL = false;

static cl_context createOpenCLContext() {
    if (initializedOpenCL) {
        throw std::runtime_error("OpenCL context is already initialized.");
    }

    cl_int err;

    // 1. Check available platforms
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        throw std::runtime_error("Failed to find any OpenCL platforms.");
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get OpenCL platform IDs.");
    }

    // 2. Check available devices, select the first GPU device
    cl_platform_id selectedPlatform = platforms[0]; // select the first platform
    cl_uint numDevices;
    err = clGetDeviceIDs(
        selectedPlatform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices
    );
    if (err != CL_SUCCESS || numDevices == 0) {
        throw std::runtime_error("Failed to find any OpenCL devices.");
    }

    std::vector<cl_device_id> devices(numDevices);
    err = clGetDeviceIDs(
        selectedPlatform,
        CL_DEVICE_TYPE_GPU,
        numDevices,
        devices.data(),
        nullptr
    );
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get OpenCL device IDs.");
    }

    // 3. Create OpenCL context
    cl_context_properties properties[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)selectedPlatform, 0
    };

    cl_context context =
        clCreateContext(properties, 1, &devices[0], nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL context.");
    }
    initializedOpenCL = true;
    return context;
}



DLManagedTensor *createDLPackTensorFromOpenCL(
    cl_context context, IOSurfaceRef ioSurface, size_t width, size_t height
) {
    // create opencl image
    cl_image_format format = {
        CL_RGBA,      // rgba channel order
        CL_UNORM_INT8 // 8-bit unsigned normalized integer
    };

    cl_int errcode;
    cl_mem clBuffer = clCreateImageFromIOSurface2DAPPLE(
        context, CL_MEM_READ_WRITE, &format, width, height, ioSurface, &errcode
    );

    if (!clBuffer || errcode != CL_SUCCESS) {
        throw std::runtime_error(
            "Failed to create OpenCL image from IOSurface: " +
            std::to_string(errcode)
        );
    }

    DLManagedTensor *tensor =
        (DLManagedTensor *)malloc(sizeof(DLManagedTensor));

    tensor->dl_tensor.data =
        reinterpret_cast<void *>(clBuffer); // set cl_mem as data
    tensor->dl_tensor.ndim = 3;             // 3D tensor (H x W x C)
    tensor->dl_tensor.shape = (int64_t *)malloc(3 * sizeof(int64_t));
    tensor->dl_tensor.shape[0] = height;
    tensor->dl_tensor.shape[1] = width;
    tensor->dl_tensor.shape[2] = 4;      // RGBA
    tensor->dl_tensor.strides = nullptr; // Dense layout
    tensor->dl_tensor.byte_offset = 0;

    tensor->dl_tensor.dtype = {kDLUInt, 8, 1}; // Unsigned 8-bit integer
    tensor->dl_tensor.device = {kDLOpenCL, 0}; // OpenCL device

    // Set memory deleter
    tensor->manager_ctx = new cl_mem{clBuffer}; // cl_mem context
    tensor->deleter = [](DLManagedTensor *self) {
        cl_mem *buffer = reinterpret_cast<cl_mem *>(self->manager_ctx);
        clReleaseMemObject(*buffer); // OpenCL release
        delete buffer;

        free(self->dl_tensor.shape); // release shape
        free(self);                  // DLManagedTensor release
    };

    return tensor;
}

#else

#if USE_CUSTOM_DL_PACK_TENSOR
#include <ATen/ATen.h>
Tensor tensor_fromDLPack(DLManagedTensor* dlMTensor) {
  auto deleter_with_gil = [dlMTensor](void*) {
    if (dlMTensor->deleter) {
      pybind11::gil_scoped_acquire gil;
      dlMTensor->deleter(dlMTensor);
    }
  };

  // atensor steals the ownership of the underlying storage. It also passes a
  // destructor function that will be called when the underlying storage goes
  // out of scope. When the destructor is called, the dlMTensor is destructed
  // too.
  auto atensor = fromDLPack(dlMTensor);

  // Make sure this capsule will never be used again.
  PyCapsule_SetName(data, "used_dltensor");

  // It is possible that the call to at::fromDLPack is the very first
  // call to create a Tensor in PyTorch. If so, then _lazy_init has
  // not been called, and the attempt to call createPyObject will fail
  // because cuda ATen types have not been registered in Python yet.
  // so if we have a cuda tensor, then we need to make sure
  // we have called _lazy_init here
  // maybe_initialize_device(atensor.device()); : MPS는 해당안함
  return atensor;
}

Tensor fromDLPack(DLManagedTensor* src, std::function<void(void*)> deleter) {
  Device device = at::Device(DeviceType::Metal, static_cast<c10::DeviceIndex>(0));;
  ScalarType stype = toScalarType(src->dl_tensor.dtype);
  if (!src->dl_tensor.strides) {
    return at::from_blob(
        src->dl_tensor.data,
        IntArrayRef(src->dl_tensor.shape, src->dl_tensor.ndim),
        std::move(deleter),
        at::device(device).dtype(stype),
        {device});
  }
  return at::from_blob(
      src->dl_tensor.data,
      IntArrayRef(src->dl_tensor.shape, src->dl_tensor.ndim),
      IntArrayRef(src->dl_tensor.strides, src->dl_tensor.ndim),
      deleter,
      at::device(device).dtype(stype),
      {device});
}


#endif

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

    // Set memory deleter
    tensor->deleter = deleteDLManagedTensor;
    return tensor;
}
#endif

DLManagedTensor *
mtl_tensor_from_mach_port(unsigned int machPort, int width, int height) {
    IOSurfaceRef ioSurface = getIOSurfaceFromMachPort((mach_port_t)machPort);
    if (!ioSurface) {
        throw std::runtime_error("Failed to initialize IOSurface");
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        throw std::runtime_error("Failed to create Metal device");
    }

    #if USE_OPENCL_DL_PACK_TENSOR
    cl_context context = createOpenCLContext();
    DLManagedTensor *tensor = createDLPackTensorFromOpenCL(context, ioSurface, width, height);
    #else
    DLManagedTensor *tensor = createDLPackTensor(ioSurface, width, height);
    #endif

    return tensor;
}
