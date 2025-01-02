#include <gtest/gtest.h>
#include "ipc.h"

TEST(IPCModuleTest, AddFunction) {
    EXPECT_EQ(add(2, 3), 5);
}

TEST(IPCModuleTest, ExceptionHandling) {
    EXPECT_THROW(safe_divide(1, 0), std::invalid_argument);
}

