#!/bin/bash
# Build script for Neural Network Library (Linux/Mac)

echo "===================================="
echo "Building Neural Network Library"
echo "===================================="

# Create build directory if it doesn't exist
mkdir -p build

# Navigate to build directory
cd build

# Configure with CMake
echo ""
echo "Configuring project with CMake..."
cmake ..

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: CMake configuration failed!"
    echo "Make sure CMake is installed and in your PATH."
    exit 1
fi

# Build the project
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

cd ..
