# Google Test
# https://google.github.io/googletest/quickstart-cmake.html
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(CUDAToolkit QUIET)
find_package(Python3 REQUIRED COMPONENTS Development)

# Build Test directory
add_executable(test_ipc test_ipc.cpp)

target_include_directories(
    test_ipc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../pybind11/include
    ${Python3_INCLUDE_DIRS}
)

if(APPLE)
    set(CMAKE_PREFIX_PATH "/usr/local/libtorch")
    find_package(Torch QUIET)
    if(Torch_FOUND)
        message(STATUS "Torch is available")
        add_definitions(-DHAS_TORCH)
    else()
        message(STATUS "Torch is not available in APPLE")
    endif()
else()
    message(STATUS "Torch is not available")
endif()

# Set up platform-specific logic
set(
    LINK_LIBRARIES
    Python3::Python
    GTest::gtest_main
    craftground
)

if(UNIX)
    if(APPLE)
        # macOS
        message(STATUS "Detected macOS")
        list(APPEND LINK_LIBRARIES
            "-lobjc"
            "-framework Metal"
            "-framework CoreGraphics"
            "-framework IOSurface"
            "-framework Foundation"
        )
        if(Torch_FOUND)
            set(LIBTORCH_DIR ${CMAKE_SOURCE_DIR}/src/cpp/libtorch)
            # target_link_libraries(test_ipc PRIVATE
            #     ${LIBTORCH_DIR}/libtorch_cpu_minimal.a
            #     ${LIBTORCH_DIR}/libc10.a
            # )
            list(APPEND LINK_LIBRARIES "${TORCH_LIBRARIES}")
        endif()
    else()
        # Ubuntu Linux
        message(STATUS "Detected Ubuntu Linux")
        if(CUDAToolkit_FOUND)
            # Ubuntu with CUDA
            message(STATUS "CUDA is available in Ubuntu")
            target_include_directories(test_ipc PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
            list(APPEND LINK_LIBRARIES CUDA::cudart CUDA::cudart_static rt)
        else()
            # Ubuntu without CUDA
            list(APPEND LINK_LIBRARIES rt)
        endif()
    endif()
elseif(WIN32)
    # Windows
    message(STATUS "Detected Windows")
    if(CUDAToolkit_FOUND)
        # Windows with CUDA
        message(STATUS "CUDA is available in Windows")
        list(APPEND LINK_LIBRARIES CUDA::cudart)
    endif()
endif()

# Apply the libraries to the target
target_link_libraries(test_ipc PRIVATE ${LINK_LIBRARIES})

# Ensure proper compile options
target_compile_options(test_ipc PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1)

# Integrate with CTest
include(GoogleTest)
gtest_discover_tests(test_ipc)
