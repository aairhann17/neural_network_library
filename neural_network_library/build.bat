@echo off
REM Build script for Neural Network Library (Windows).
REM
REM The script configures the project with CMake, stops on failure, and places
REM the resulting binaries in the Visual Studio Release output directory.

echo ====================================
echo Building Neural Network Library
echo ====================================

REM Create the build directory if it does not already exist.
if not exist build mkdir build

REM Move into the build directory so generated files stay out of the source tree.
cd build

REM Generate the Visual Studio or Makefile project files.
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

REM Compile the Release configuration of the project.
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
