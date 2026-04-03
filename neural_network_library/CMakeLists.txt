cmake_minimum_required(VERSION 3.15)

# Name the project and expose a version so downstream consumers can inspect it.
project(NeuralNetCpp VERSION 0.1.0)

option(BUILD_DOCS "Build API documentation with Doxygen" OFF)

# Require modern C++ support across all targets in this repository.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile_commands.json for tooling such as clangd.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    # Default to Release so example binaries are optimized in typical local builds.
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler warnings
if(MSVC)
    # /W4 enables a strong warning baseline on Visual C++.
    add_compile_options(/W4)
else()
    # Match the MSVC warning intent on GCC/Clang.
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Optimization flags for Release
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        add_compile_options(/O2)
    else()
        # Fast-math and native tuning favor local performance over portability.
        add_compile_options(-O3 -march=native -ffast-math)
    endif()
endif()

# Probe for OpenMP so the library can opt into parallel execution when available.
find_package(OpenMP)

if(BUILD_DOCS)
    find_package(Doxygen QUIET)
    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "BUILD_DOCS is ON but Doxygen was not found")
    endif()

    set(DOXYGEN_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/docs")
    set(DOXYGEN_OUTPUT_DIRECTORY "${DOXYGEN_GENERATED_DIR}")
    set(DOXYGEN_INPUT_DIRS
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/examples"
        "${DOXYGEN_MAINPAGE}"
    )
    set(DOXYGEN_MAINPAGE "${CMAKE_CURRENT_SOURCE_DIR}/../README.md")
    string(REPLACE ";" " " DOXYGEN_INPUT "${DOXYGEN_INPUT_DIRS}")
    set(DOXYFILE_IN "${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in")
    set(DOXYFILE_OUT "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")

    configure_file("${DOXYFILE_IN}" "${DOXYFILE_OUT}" @ONLY)

    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} "${DOXYFILE_OUT}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()

# Build the reusable static/shared library target from the core implementation.
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
    # Only link OpenMP when the current toolchain actually provides it.
    target_link_libraries(neuralnet PUBLIC OpenMP::OpenMP_CXX)
endif()

# Examples
add_executable(xor_example examples/xor_example.cpp)
target_link_libraries(xor_example neuralnet)

add_executable(regression_example examples/regression_example.cpp)
target_link_libraries(regression_example neuralnet)

# Enable testing
enable_testing()

# Conditionally register tests so the project still configures cleanly when the
# optional tests directory is not present in the checkout.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_tensor.cpp")
    add_executable(test_tensor tests/test_tensor.cpp)
    target_link_libraries(test_tensor neuralnet)
    add_test(NAME TensorTests COMMAND test_tensor)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_layers.cpp")
    add_executable(test_layers tests/test_layers.cpp)
    target_link_libraries(test_layers neuralnet)
    add_test(NAME LayerTests COMMAND test_layers)
endif()
