cmake_minimum_required(VERSION 3.15)
project(NeuralNetCpp VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Optimization flags for Release
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        add_compile_options(/O2)
    else()
        add_compile_options(-O3 -march=native -ffast-math)
    endif()
endif()

# Find OpenMP (for parallelization later)
find_package(OpenMP)

# Main library
add_library(neuralnet
    src/tensor.cpp
    src/activations.cpp
    src/losses.cpp
    src/module.cpp
    src/optimizer.cpp
    src/sequential.cpp
)

target_include_directories(neuralnet PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(neuralnet PUBLIC OpenMP::OpenMP_CXX)
endif()

# Examples
add_executable(xor_example examples/xor_example.cpp)
target_link_libraries(xor_example neuralnet)

add_executable(regression_example examples/regression_example.cpp)
target_link_libraries(regression_example neuralnet)

# Enable testing
enable_testing()

# Tests (if test file exists)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tensor.cpp")
    add_executable(test_tensor tests/test_tensor.cpp)
    target_link_libraries(test_tensor neuralnet)
endif()
add_test(NAME TensorTests COMMAND test_tensor)

add_executable(test_layers tests/test_layers.cpp)
target_link_libraries(test_layers neuralnet)
add_test(NAME LayerTests COMMAND test_layers)

include\tensor.hpp
    Tensor& operator*=(const Tensor& other);