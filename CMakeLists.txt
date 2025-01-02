cmake_minimum_required(VERSION 3.15...3.29)
project(craftground LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


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
if(APPLE AND Torch_FOUND)
    message(STATUS "Torch is available")
    add_definitions(-DHAS_TORCH)
else()
    message(STATUS "Torch is not available")
endif()

# Collect source files for the module
set(CRAFTGROUND_PY_SOURCES src/cpp/ipc.cpp)

if (APPLE)
    # Add Apple-specific source files
    list(APPEND CRAFTGROUND_PY_SOURCES src/cpp/ipc_apple.mm src/cpp/ipc_apple_torch.cpp)

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
        set(LIBTORCH_DIR ${CMAKE_SOURCE_DIR}/src/cpp/libtorch)
        target_link_libraries(craftground_native PRIVATE
           ${LIBTORCH_DIR}/libtorch_cpu_minimal.a
            ${LIBTORCH_DIR}/libc10.a
        )
        target_include_directories(craftground_native PRIVATE "${TORCH_INCLUDE_DIRS}")
        target_compile_definitions(craftground_native PRIVATE HAS_TORCH)
        target_compile_options(craftground_native PRIVATE -ffunction-sections -fdata-sections)
        target_link_options(craftground_native PRIVATE -Wl,-dead_strip)
        # add_custom_command(
        # TARGET craftground_native POST_BUILD
        # COMMAND install_name_tool -change @rpath/libc10.dylib @loader_path/libc10.dylib $<TARGET_FILE:craftground_native>
        # COMMAND install_name_tool -change @rpath/libtorch.dylib @loader_path/libtorch.dylib $<TARGET_FILE:craftground_native>
        # COMMAND install_name_tool -change @rpath/libtorch_cpu.dylib @loader_path/libtorch_cpu.dylib $<TARGET_FILE:craftground_native>
        # COMMENT "Updating RPATH to use relative paths"
        # )

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

option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_library(craftground STATIC ${CRAFTGROUND_PY_SOURCES})
    add_subdirectory(tests)
endif()