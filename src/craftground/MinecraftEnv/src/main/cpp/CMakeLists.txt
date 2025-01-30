cmake_minimum_required(VERSION 3.28)
project(framebuffer_capturer)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)
set(CMAKE_FIND_PACKAGE_SORT_DIRECTION DEC)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG bf71a834948186f4097caa076cd2663c69a10e1e #refs/tags/1.0.1
)

FetchContent_MakeAvailable(glm)

set(CRAFGROUND_NATIVE_DEBUG $ENV{CRAFGROUND_NATIVE_DEBUG})
if(CRAFGROUND_NATIVE_DEBUG)
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
if(NOT APPLE)
    find_package(GLEW REQUIRED)
    include_directories(${GLEW_INCLUDE_DIRS})
    message(GLEW_INCLUDE_DIRS="${GLEW_INCLUDE_DIRS}")
    message(GLEW_LIBRARIES="${GLEW_LIBRARIES}")
endif()

# Find png
if(APPLE)
    set(CMAKE_FIND_FRAMEWORK NEVER)
    message("CMAKE_FIND_FRAMEWORK=${CMAKE_FIND_FRAMEWORK}")
endif()
# Hacky: Set the CMAKE_FIND_FRAMEWORK to LAST to avoid finding mismatching libs/headers
set(CMAKE_FIND_FRAMEWORK LAST)
# set(PNG_STATIC ON)
find_package(PNG 1.6 QUIET)
if(PNG_FOUND)
    include_directories(${PNG_INCLUDE_DIRS})
    message(PNG_INCLUDE_DIRS="${PNG_INCLUDE_DIRS}")
    message(PNG_LIBRARIES="${PNG_LIBRARIES}")
    add_compile_definitions(HAS_PNG)
else()
    message(WARNING "libpng not found. You won't be able to use PNG mode. To enable PNG mode, install libpng and zlib and re-run cmake.")
endif()

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA is available")
    add_definitions(-DHAS_CUDA)
else()
    message(STATUS "CUDA is not available")
endif()

# Find Boost
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

message(STATUS "Boost is now available")

# Add your source files here
set(
    SOURCE_FILES
    framebuffer_capturer.cpp
    noboost_ipc.cpp
)

message(STATUS "SOURCE_FILES=${SOURCE_FILES}")

if(APPLE)
    list(APPEND SOURCE_FILES framebuffer_capturer_apple.mm)
endif()

if(CUDAToolkit_FOUND)
    list(APPEND SOURCE_FILES framebuffer_capturer_cuda.cpp)
endif()

# Add a library target for your native library
add_library(native-lib SHARED ${SOURCE_FILES})

if(CRAFGROUND_NATIVE_DEBUG)
    message("CRAFGROUND_NATIVE_DEBUG=${CRAFGROUND_NATIVE_DEBUG}")
    target_compile_options(native-lib PRIVATE -g) # Add debug symbols
endif()

# Link with JNI and OpenGL libraries
target_link_libraries(native-lib ${JNI_LIBRARIES} ${OPENGL_LIBRARIES} glm::glm)
# target_link_libraries(native-lib Boost::system Boost::thread Boost::interprocess)
# target_include_directories(native-lib PRIVATE ${Boost_INCLUDE_DIRS})

if(PNG_FOUND)
    target_link_libraries(native-lib ${PNG_LIBRARIES} ${ZLIB_LIBRARIES})
endif()

if(NOT APPLE)
    target_link_libraries(native-lib ${GLEW_LIBRARIES})
endif()

if(APPLE)
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
    target_link_libraries(native-lib CUDA::cudart)
endif()

# If you are using additional libraries, like GLFW for OpenGL context creation,
# you can find and link them here as well.
# For example:
# find_package(glfw3 REQUIRED)
# target_link_libraries(native-lib glfw)