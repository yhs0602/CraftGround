cmake_minimum_required(VERSION 3.15...3.29)
project(craftground LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(pybind11)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA is available")
    add_definitions(-DHAS_CUDA)
else()
    message(STATUS "CUDA is not available")
endif()

set(CMAKE_PREFIX_PATH "/usr/local/libtorch")
find_package(Torch QUIET)
if(Torch_FOUND)
    message(STATUS "Torch is available")
    add_definitions(-DHAS_TORCH)
else()
    message(STATUS "Torch is not available")
endif()

# Collect source files for the module
set(CRAFTGROUND_PY_SOURCES src/cpp/ipc.cpp)

if (APPLE)
    # Add Apple-specific source files
    list(APPEND CRAFTGROUND_PY_SOURCES src/cpp/ipc_apple.mm)

    # Apple-specific compile options
    set(CRAFTGROUND_PY_COMPILE_OPTIONS -fobjc-arc)
    if(Torch_FOUND)
    set(CRAFTGROUND_PY_COMPILE_OPTIONS -DHAS_TORCH)
    endif()
elseif(CUDAToolkit_FOUND)
    # Add CUDA-specific source files
    list(APPEND CRAFTGROUND_PY_SOURCES src/cpp/ipc_cuda.cpp)

    # CUDA-specific compile options
    set(CRAFTGROUND_PY_COMPILE_OPTIONS -DHAS_CUDA)
endif()

# Add the module
pybind11_add_module(craftground_native ${CRAFTGROUND_PY_SOURCES})

if(APPLE)
    if(Torch_FOUND)
        target_link_libraries(craftground_native PRIVATE "${TORCH_LIBRARIES}")
        target_include_directories(craftground_native PRIVATE "${TORCH_INCLUDE_DIRS}")
    endif()
elseif(CUDAToolkit_FOUND)
    target_include_directories(craftground_native PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(craftground_native PRIVATE ${CUDAToolkit_LIBRARIES})
endif()


target_include_directories(craftground_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp)

target_compile_options(craftground_native PRIVATE ${CRAFTGROUND_PY_COMPILE_OPTIONS})

target_compile_definitions(
    craftground_native
    PRIVATE VERSION_INFO=${PRIVATE_VERSION_INFO}
)

install(TARGETS craftground_native LIBRARY DESTINATION craftground)