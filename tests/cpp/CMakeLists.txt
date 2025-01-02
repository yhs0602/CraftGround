# Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

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
    Python3::Python
    GTest::gtest_main
    CUDA::cudart
    CUDA::cudart_static
    craftground
)
target_compile_options(test_ipc PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1)

# Integrate with CTest
include(GoogleTest)
gtest_discover_tests(test_ipc)

