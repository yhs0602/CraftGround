#include <gtest/gtest.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include "ipc.h"

#ifdef __APPLE__
TEST(IPCModuleTest, AddFunction) {
    EXPECT_THROW(initialize_from_mach_port(2, 3, 4), std::runtime_error);
}
#else

TEST(IPCModuleTest, AddFunction) {
    EXPECT_EQ(initialize_from_mach_port(2, 3, 4), py::none());
}
#endif
// TEST(IPCModuleTest, ExceptionHandling) {
// EXPECT_EQ
//     EXPECT_THROW(
//         mtl_tensor_from_cuda_mem_handle("", 0, 2), std::invalid_argument
//     );
// }
