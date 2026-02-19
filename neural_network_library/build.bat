@echo off
REM Build script for Neural Network Library (Windows)

echo ====================================
echo Building Neural Network Library
echo ====================================

REM Create build directory if it doesn't exist
if not exist build mkdir build

REM Navigate to build directory
cd build

REM Configure with CMake
echo.
echo Configuring project with CMake...
cmake ..

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: CMake configuration failed!
    echo Make sure CMake is installed and in your PATH.
    pause
    exit /b 1
)

REM Build the project
echo.
echo Building project...
cmake --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo ====================================
echo Build completed successfully!
echo ====================================
echo.
echo Executables are in: build\Release\
echo   - xor_example.exe
echo   - regression_example.exe
echo.

cd ..
pause
