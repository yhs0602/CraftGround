#include <gtest/gtest.h>
#include <pybind11/pybind11.h>
#include "ipc.h"

TEST(IPCModuleTest, AddFunction) {
    EXPECT_EQ(initialize_from_mach_port(2, 3, 4), py::none());
}

// TEST(IPCModuleTest, ExceptionHandling) {
//     EXPECT_THROW(
//         mtl_tensor_from_cuda_mem_handle("", 0, 2), std::invalid_argument
//     );
// }
