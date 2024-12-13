cmake_minimum_required(VERSION 3.15...3.29)
project(craftground_py LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(pybind11)
pybind11_add_module(craftground_py src/cpp/ipc_apple.cpp src/cpp/ipc_apple.mm)

target_compile_options(craftground_py PRIVATE -fobjc-arc)
target_include_directories(craftground_py PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp)
target_compile_definitions(
    craftground_py
    PRIVATE VERSION_INFO=${PRIVATE_VERSION_INFO}
)
