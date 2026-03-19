#ifndef __IPC_APPLE_H__
#define __IPC_APPLE_H__

#define USE_CUSTOM_DL_PACK_TENSOR 1

py::object
mtl_tensor_from_mach_port(unsigned int machPort, int width, int height);
py::object normalize_apple_mtl_tensor_impl(py::object tensor);

#endif // __IPC_APPLE_H__
