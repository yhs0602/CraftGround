#include <ATen/DLConvertor.h>
#include <torch/extension.h>
#include "ipc_apple.h"

#if USE_CUSTOM_DL_PACK_TENSOR
at::Tensor
fromDLPack(DLManagedTensor *src, std::function<void(void *)> deleter) {
    at::Device device =
        at::Device(at::DeviceType::MPS, static_cast<c10::DeviceIndex>(0));
    at::ScalarType stype = at::toScalarType(src->dl_tensor.dtype);
    if (!src->dl_tensor.strides) {
        return at::from_blob(
            src->dl_tensor.data,
            at::IntArrayRef(src->dl_tensor.shape, src->dl_tensor.ndim),
            std::move(deleter),
            at::device(device).dtype(stype),
            {device}
        );
    }
    return at::from_blob(
        src->dl_tensor.data,
        at::IntArrayRef(src->dl_tensor.shape, src->dl_tensor.ndim),
        at::IntArrayRef(src->dl_tensor.strides, src->dl_tensor.ndim),
        deleter,
        at::device(device).dtype(stype),
        {device}
    );
}

// patch: torch/csrc/utils/tensor_new.cpp
// torch/csrc/Module.cpp
PyObject *torchTensorFromDLPack(DLManagedTensor *dlMTensor) {
    auto deleter_with_gil = [dlMTensor](void *) {
        if (dlMTensor->deleter) {
            pybind11::gil_scoped_acquire gil;
            dlMTensor->deleter(dlMTensor);
        }
    };

    // atensor steals the ownership of the underlying storage. It also passes a
    // destructor function that will be called when the underlying storage goes
    // out of scope. When the destructor is called, the dlMTensor is destructed
    // too.
    auto atensor = fromDLPack(dlMTensor, deleter_with_gil);

    // Make sure this capsule will never be used again.
    // We do not make python dltensor object as a capsule so we need not call it
    //   PyCapsule_SetName(python_dltensor_object, "used_dltensor");

    // It is possible that the call to at::fromDLPack is the very first
    // call to create a Tensor in PyTorch. If so, then _lazy_init has
    // not been called, and the attempt to call createPyObject will fail
    // because cuda ATen types have not been registered in Python yet.
    // so if we have a cuda tensor, then we need to make sure
    // we have called _lazy_init here
    // maybe_initialize_device(atensor.device()); : No need to call this
    // function in metal
    return THPVariable_Wrap(atensor);
}

#endif