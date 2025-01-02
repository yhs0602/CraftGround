# Google Test
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Build Test directory
add_executable(test_ipc test_ipc.cpp)
target_link_libraries(test_ipc PRIVATE craftground GTest::GTest GTest::Main)

# Integrate with CTest
add_test(NAME CppTests COMMAND test_ipc)

