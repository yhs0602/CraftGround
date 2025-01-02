# Google Test
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

find_package(CUDAToolkit REQUIRED)

find_package(Python3 REQUIRED COMPONENTS Development)

# Build Test directory
add_executable(test_ipc test_ipc.cpp)

target_include_directories(
    test_ipc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../pybind11/include
    ${Python3_INCLUDE_DIRS}
)
target_include_directories(test_ipc PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
message(STATUS "CUDA is available in tests")
message(STATUS "CUDa toolkit libraries: ${CUDAToolkit_LIBRARY_DIR}")
target_link_libraries(
    test_ipc PRIVATE
    GTest::GTest
    GTest::Main
    Python3::Python
    CUDA::cudart
    CUDA::cudart_static
    craftground
)
target_compile_options(test_ipc PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1)


# Integrate with CTest
add_test(NAME CppTests COMMAND test_ipc)

