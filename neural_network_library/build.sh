#!/bin/bash
# Build script for Neural Network Library (Linux/Mac).
#
# The script configures the project with CMake, aborts on configuration or build
# errors, and leaves the resulting binaries in the local build directory.

echo "===================================="
echo "Building Neural Network Library"
echo "===================================="

# Create the out-of-source build directory when it is missing.
mkdir -p build

# Enter the build directory so CMake artifacts stay isolated from sources.
cd build

# Generate platform-specific build files.
echo ""
echo "Configuring project with CMake..."
cmake ..

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: CMake configuration failed!"
    echo "Make sure CMake is installed and in your PATH."
    exit 1
fi

# Compile the project using all available logical CPUs when possible.
echo ""
echo "Building project..."
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Build failed!"
    exit 1
fi

echo ""
echo "===================================="
echo "Build completed successfully!"
echo "===================================="
echo ""
echo "Executables are in: build/"
echo "  - xor_example"
echo "  - regression_example"
echo ""

# Return to the repository root for convenience before exiting.
cd ..
