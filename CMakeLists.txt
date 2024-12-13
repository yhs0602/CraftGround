cmake_minimum_required(VERSION 3.15...3.29)
project(craftground_py LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(pybind11)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA is available")
    add_definitions(-DHAS_CUDA)
else()
    message(STATUS "CUDA is not available")
endif()

# Collect source files for the module
set(CRAFTGROUND_PY_SOURCES src/cpp/ipc.cpp)

if (APPLE)
    # Add Apple-specific source files
    list(APPEND CRAFTGROUND_PY_SOURCES src/cpp/ipc_apple.mm)

    # Apple-specific compile options
    set(CRAFTGROUND_PY_COMPILE_OPTIONS -fobjc-arc)
elseif(CUDAToolkit_FOUND)
    # Add CUDA-specific source files
    list(APPEND CRAFTGROUND_PY_SOURCES src/cpp/ipc_cuda.cpp)

    # CUDA-specific compile options
    set(CRAFTGROUND_PY_COMPILE_OPTIONS -DHAS_CUDA)
endif()

# Add the module
pybind11_add_module(craftground_py ${CRAFTGROUND_PY_SOURCES})

target_include_directories(craftground_py PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp)

target_compile_options(craftground_py PRIVATE ${CRAFTGROUND_PY_COMPILE_OPTIONS})

target_compile_definitions(
    craftground_py
    PRIVATE VERSION_INFO=${PRIVATE_VERSION_INFO}
)
