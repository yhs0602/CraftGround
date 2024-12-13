#include "ipc_apple.h"
#include "../../pybind11/include/pybind11/pybind11.h"

namespace py = pybind11;
py::object initialize_from_mach_port(int machPort, int width, int height) {
    DLManagedTensor* tensor = mtl_tensor_from_mach_port(machPort, width, height);
    return py::reinterpret_steal<py::object>(
        PyCapsule_New(tensor, "dltensor", [](PyObject* capsule) {
            DLManagedTensor* tensor = (DLManagedTensor*)PyCapsule_GetPointer(capsule, "dltensor");
            tensor->deleter(tensor);
        })
    );
}
