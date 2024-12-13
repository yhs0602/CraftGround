cmake_minimum_required(VERSION 3.10)
project(framebuffer_capturer)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
SET(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)
SET(CMAKE_FIND_PACKAGE_SORT_DIRECTION DEC)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CRAFGROUND_NATIVE_DEBUG $ENV{CRAFGROUND_NATIVE_DEBUG})
if (CRAFGROUND_NATIVE_DEBUG)
    message("CRAFGROUND_NATIVE_DEBUG=${CRAFGROUND_NATIVE_DEBUG}")
    set(CMAKE_BUILD_TYPE Debug) # Set default build type to Debug
else()
    message("CRAFGROUND_NATIVE_DEBUG not set")
endif()


# Find JNI
find_package(Java COMPONENTS Development)
find_package(JNI REQUIRED)
include_directories(${JNI_INCLUDE_DIRS})

# Find OpenGL
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})
if (NOT APPLE)
    find_package(GLEW REQUIRED)
    include_directories(${GLEW_INCLUDE_DIRS})
    message(GLEW_INCLUDE_DIRS="${GLEW_INCLUDE_DIRS}")
    message(GLEW_LIBRARIES="${GLEW_LIBRARIES}")
endif()

# Find png
if (APPLE)
  set(CMAKE_FIND_FRAMEWORK NEVER)
  message("CMAKE_FIND_FRAMEWORK=${CMAKE_FIND_FRAMEWORK}")
endif()
# Hacky: Set the CMAKE_FIND_FRAMEWORK to LAST to avoid finding mismatching libs/headers
set(CMAKE_FIND_FRAMEWORK LAST)
# set(PNG_STATIC ON)
find_package(PNG 1.6 REQUIRED)
include_directories(${PNG_INCLUDE_DIRS})
message(PNG_INCLUDE_DIRS="${PNG_INCLUDE_DIRS}")
message(PNG_LIBRARIES="${PNG_LIBRARIES}")
# find_library(LIBPNG_STATIC libpng16.a)
#if(NOT LIBPNG_STATIC)
#  message(FATAL_ERROR "libpng static library not found")
#else()
#  message(LIBPNG_STATIC="${LIBPNG_STATIC}")
#endif()
# Find zlib
# find_package(ZLIB REQUIRED)
# include_directories(${ZLIB_INCLUDE_DIRS})
# find CUDA

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA is available")
    add_definitions(-DHAS_CUDA)
else()
    message(STATUS "CUDA is not available")
endif()


# Add your source files here
set(SOURCE_FILES
    framebuffer_capturer.cpp
)

if(APPLE)
    list(APPEND SOURCE_FILES framebuffer_capturer_apple.mm)
endif()

# Add a library target for your native library
add_library(native-lib SHARED ${SOURCE_FILES})

if (CRAFGROUND_NATIVE_DEBUG)
    message("CRAFGROUND_NATIVE_DEBUG=${CRAFGROUND_NATIVE_DEBUG}")
    target_compile_options(native-lib PRIVATE -g) # Add debug symbols
endif()

# Link with JNI and OpenGL libraries
target_link_libraries(native-lib ${JNI_LIBRARIES} ${OPENGL_LIBRARIES} ${PNG_LIBRARIES} ${ZLIB_LIBRARIES})
if (NOT APPLE)
    target_link_libraries(native-lib ${GLEW_LIBRARIES})
endif()

if (APPLE)
    target_link_libraries(
        native-lib
        "-framework Metal"
        "-framework CoreGraphics"
        "-framework IOSurface"
        "-framework Foundation"
        "-lobjc"
    )
endif()

if(CUDAToolkit_FOUND)
    target_link_libraries(native-lib PRIVATE CUDA::cudart)
endif()

# If you are using additional libraries, like GLFW for OpenGL context creation,
# you can find and link them here as well.
# For example:
# find_package(glfw3 REQUIRED)
# target_link_libraries(native-lib glfw)