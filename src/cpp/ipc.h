#ifndef __IPC_H__
#define __IPC_H__
#include "../../pybind11/include/pybind11/pybind11.h"
#include "../../pybind11/include/pybind11/stl.h"
#include "../../pybind11/include/pybind11/pytypes.h"

namespace py = pybind11;
py::object initialize_from_mach_port(unsigned int machPort, int width, int height);
py::object
mtl_tensor_from_cuda_mem_handle(const char* cuda_ipc_handle, int width, int height);

#endif