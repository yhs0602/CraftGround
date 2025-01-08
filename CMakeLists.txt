cmake_minimum_required(VERSION 3.15...3.29)
project(craftground LANGUAGES CXX)
include(GNUInstallDirs)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


add_subdirectory(pybind11)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA is available: ${CUDAToolkit_VERSION}(${CUDAToolkit_INCLUDE_DIRS}) (${CUDAToolkit_LIBRARY_DIR}) (${CUDAToolkit_LIBRARY_ROOT})")
    add_definitions(-DHAS_CUDA)
else()
    message(STATUS "CUDA is not available")
endif()


if(APPLE)
    set(CMAKE_PREFIX_PATH "/usr/local/libtorch")
    find_package(Torch QUIET)
    if(Torch_FOUND)
        message(STATUS "Torch is available: libraries: ${TORCH_LIBRARIES}")
        add_definitions(-DHAS_TORCH)
        list(APPEND TORCH_LIBRARIES_LIST ${TORCH_LIBRARIES})

        message(STATUS "TORCH_LIBRARIES_LIST: ${TORCH_LIBRARIES_LIST}")
        set(LIBTORCH_PATHS "")
        foreach(LIB ${TORCH_LIBRARIES_LIST})
            if(LIB MATCHES "/[^/]+\\.(dylib|a)$") # .dylib 또는 .a로 끝나는 항목만 처리
                string(REGEX REPLACE "(.*)/[^/]+\\.(dylib|a)$" "\\1" DIR_PATH "${LIB}")
                list(APPEND LIBTORCH_DIRS ${DIR_PATH})
            endif()
        endforeach()
        message(STATUS "Extracted LIBTORCH_DIRS: ${LIBTORCH_DIRS}")
        list(REMOVE_DUPLICATES LIBTORCH_DIRS)

        message(STATUS "LIBTORCH_DIRS: ${LIBTORCH_DIRS}")

        set(LIBOMP_PATH "${LIBTORCH_DIRS}/libomp.dylib")
        set(LIBC10_PATH "${LIBTORCH_DIRS}/libc10.dylib")
        set(LIBTORCH_CPU_PATH "${LIBTORCH_DIRS}/libtorch_cpu.dylib")
        set(LIBTORCH_PYTHON_PATH "${LIBTORCH_DIRS}/libtorch_python.dylib")
        message(STATUS "LIBOMP_PATH: ${LIBOMP_PATH}")
        message(STATUS "LIBC10_PATH: ${LIBC10_PATH}")
        message(STATUS "LIBTORCH_CPU_PATH: ${LIBTORCH_CPU_PATH}")
        message(STATUS "LIBTORCH_PYTHON_PATH: ${LIBTORCH_PYTHON_PATH}")
        if(NOT EXISTS "${LIBOMP_PATH}")
            message(FATAL_ERROR "LIBOMP_PATH does not exist: ${LIBOMP_PATH}")
        endif()

        if(NOT EXISTS "${LIBC10_PATH}")
            message(FATAL_ERROR "LIBC10_PATH does not exist: ${LIBC10_PATH}")
        endif()

        if(NOT EXISTS "${LIBTORCH_CPU_PATH}")
            message(FATAL_ERROR "LIBTORCH_CPU_PATH does not exist: ${LIBTORCH_CPU_PATH}")
        endif()
    else()
        message(STATUS "Torch is not available in APPLE")
    endif()
else()
    message(STATUS "Torch is not available")
endif()

# Collect source files for the module
set(CRAFTGROUND_PY_SOURCES src/cpp/ipc.cpp)

if(APPLE)
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
        target_include_directories(craftground_native PRIVATE "${TORCH_INCLUDE_DIRS}")
        target_compile_definitions(craftground_native PRIVATE HAS_TORCH)
        target_compile_options(craftground_native PRIVATE -ffunction-sections -fdata-sections)
        target_link_libraries(craftground_native PRIVATE ${LIBTORCH_CPU_PATH} ${LIBC10_PATH} ${LIBOMP_PATH} ${LIBTORCH_PYTHON_PATH})
        target_link_options(craftground_native PRIVATE -Wl,-dead_strip)
    endif()
    target_link_libraries(craftground_native PRIVATE "-lobjc")
elseif(CUDAToolkit_FOUND)
    target_include_directories(craftground_native PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    if(WIN32)
        message(STATUS "CUDA is available on windows: ${CUDAToolkit_LIBRARY_DIR}")
        target_link_libraries(
            craftground_native PRIVATE
            ${CUDAToolkit_LIBRARY_DIR}/cudart.lib
            ${CUDAToolkit_LIBRARY_DIR}/cudadevrt.lib
            ${CUDAToolkit_LIBRARY_DIR}/cudart_static.lib
            ${CUDAToolkit_LIBRARY_DIR}/cuda.lib
        )
    else()
        # On Linux, use the CUDA targets provided by CMake
        target_link_libraries(craftground_native PRIVATE CUDA::cudart CUDA::cudart_static)
    endif()
endif()


target_include_directories(craftground_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp)

target_compile_options(craftground_native PRIVATE ${CRAFTGROUND_PY_COMPILE_OPTIONS})

target_compile_definitions(
    craftground_native
    PRIVATE VERSION_INFO=${PRIVATE_VERSION_INFO}
)

install(TARGETS craftground_native LIBRARY DESTINATION craftground)

option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_library(craftground STATIC ${CRAFTGROUND_PY_SOURCES})
    set(Python_FIND_VIRTUALENV FIRST)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    target_include_directories(
        craftground PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/pybind11/include
        ${Python3_INCLUDE_DIRS}
    )
    if(APPLE)
        if(Torch_FOUND)
            set(LIBTORCH_DIR ${CMAKE_SOURCE_DIR}/src/cpp/libtorch)
            message(STATUS "Torch is available in tests: ${TORCH_LIBRARIES}")
            target_include_directories(craftground PUBLIC "${TORCH_INCLUDE_DIRS}")
            target_compile_definitions(craftground PUBLIC HAS_TORCH)
            target_compile_options(craftground PUBLIC -ffunction-sections -fdata-sections -fobjc-arc)
            target_link_options(craftground PUBLIC -Wl,-dead_strip)
            message(STATUS "Python libraries: ${Python3_LIBRARIES}")
            target_link_libraries(craftground PUBLIC ${LIBTORCH_CPU_PATH} ${LIBC10_PATH} ${LIBOMP_PATH} ${LIBTORCH_PYTHON_PATH} Python3::Python)
        endif()
        target_link_libraries(craftground PRIVATE "-lobjc")
    elseif(CUDAToolkit_FOUND)
        message(STATUS "CUDA is available in tests")
        target_include_directories(craftground PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
        if(WIN32)
            target_link_libraries(
                craftground_native PRIVATE
                ${CUDAToolkit_LIBRARY_DIR}/cudart.lib
                # ${CUDAToolkit_LIBRARY_DIR}/cudart_static.lib
                ${CUDAToolkit_LIBRARY_DIR}/cuda.lib
            )
        else()
            # On Linux, use the CUDA targets provided by CMake
            target_link_libraries(craftground_native PRIVATE CUDA::cudart CUDA::cudart_static)
        endif()
    endif()
    target_link_options(craftground PRIVATE "-Wl,--whole-archive" "-Wl,--no-whole-archive")
    target_include_directories(craftground PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp)

    target_compile_options(craftground PUBLIC ${CRAFTGROUND_PY_COMPILE_OPTIONS})

    target_compile_definitions(
        craftground
        PUBLIC VERSION_INFO=${PRIVATE_VERSION_INFO}
    )
    add_subdirectory(tests)
endif()
