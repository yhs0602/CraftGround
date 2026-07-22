cmake_minimum_required(VERSION 3.28)
project(craftground LANGUAGES CXX)
include(GNUInstallDirs)
include(FetchContent)
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


# Collect source files for the module
set(CRAFTGROUND_PY_SOURCES shared/native-ipc/ipc.cpp shared/native-ipc/ipc_noboost.cpp shared/native-ipc/print_hex.cpp)

if(APPLE)
    # Add Apple-specific source files
    list(APPEND CRAFTGROUND_PY_SOURCES shared/native-ipc/ipc_apple.mm)

    # Apple-specific compile options
    set(CRAFTGROUND_PY_COMPILE_OPTIONS -fobjc-arc)
elseif(CUDAToolkit_FOUND)
    # Add CUDA-specific source files
    list(APPEND CRAFTGROUND_PY_SOURCES shared/native-ipc/ipc_cuda.cpp)

    # CUDA-specific compile options
    set(CRAFTGROUND_PY_COMPILE_OPTIONS -DHAS_CUDA)
endif()

# Find Boost
message(STATUS "Installing Boost")
set(BOOST_INCLUDE_LIBRARIES interprocess thread system)
set(BOOST_ENABLE_CMAKE ON)

set(FETCHCONTENT_QUIET FALSE)
FetchContent_Declare(
  Boost
  GIT_REPOSITORY https://github.com/boostorg/boost.git
  GIT_TAG boost-1.87.0
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
  EXCLUDE_FROM_ALL
)
# FetchContent_MakeAvailable(Boost)

# message(STATUS "Boost is now available")
# Add the module
pybind11_add_module(craftground_native ${CRAFTGROUND_PY_SOURCES})
# target_link_libraries(craftground_native PRIVATE Boost::system Boost::thread Boost::interprocess)
# target_include_directories(craftground_native PRIVATE ${Boost_INCLUDE_DIRS})
if(APPLE)
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
if(WIN32)
    add_definitions(-D_WIN32)
    target_link_libraries(craftground_native PRIVATE kernel32.lib user32.lib Synchronization.lib)
endif()


target_include_directories(craftground_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/shared/native-ipc)

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
        target_compile_options(craftground PUBLIC -fobjc-arc)
        target_link_libraries(craftground PRIVATE "-lobjc")
    elseif(CUDAToolkit_FOUND)
        message(STATUS "CUDA is available in tests")
        target_include_directories(craftground PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
        if(WIN32)
            target_link_libraries(
                craftground PRIVATE
                ${CUDAToolkit_LIBRARY_DIR}/cudart.lib
                # ${CUDAToolkit_LIBRARY_DIR}/cudart_static.lib
                ${CUDAToolkit_LIBRARY_DIR}/cuda.lib
            )
        else()
            # On Linux, use the CUDA targets provided by CMake
            target_link_libraries(craftground PRIVATE CUDA::cudart CUDA::cudart_static)
        endif()
    endif()
    # target_link_libraries(craftground PRIVATE Boost::system Boost::thread Boost::interprocess)
    # target_include_directories(craftground PRIVATE ${Boost_INCLUDE_DIRS})
    if(WIN32)
        add_definitions(-D_WIN32)
        target_link_libraries(craftground PRIVATE kernel32.lib user32.lib Synchronization.lib)
    endif()
    target_link_options(craftground PRIVATE "-Wl,--whole-archive" "-Wl,--no-whole-archive")
    target_include_directories(craftground PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/shared/native-ipc)

    target_compile_options(craftground PUBLIC ${CRAFTGROUND_PY_COMPILE_OPTIONS})

    target_compile_definitions(
        craftground
        PUBLIC VERSION_INFO=${PRIVATE_VERSION_INFO}
    )
    add_subdirectory(tests)
endif()
