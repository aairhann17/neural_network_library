# Quick Start Guide

## Installation and Building

### Prerequisites

1. **CMake** (version 3.15 or higher)
   - Windows: Download from https://cmake.org/download/
   - Linux: `sudo apt-get install cmake` (Ubuntu/Debian) or `sudo yum install cmake` (RedHat/CentOS)
   - Mac: `brew install cmake`

2. **C++ Compiler with C++17 support**
   - Windows: Visual Studio 2017 or later (with C++ desktop development)
   - Linux: GCC 7+ or Clang 5+
   - Mac: Xcode command line tools

3. **OpenMP** (optional, for parallelization)
   - Usually comes with your compiler

### Building the Library

#### Option 1: Using the build script

**Windows:**
```cmd
build.bat
```

**Linux/Mac:**
```bash
chmod +x build.sh
./build.sh
```

#### Option 2: Manual build

```bash
# Create build directory
mkdir build
cd build

# Configure
cmake ..

# Build
cmake --build . --config Release

# On Linux/Mac with make
make -j4
```

### Running Examples

After building, run the examples:

**Windows:**
```cmd
.\build\Release\xor_example.exe
.\build\Release\regression_example.exe
```

**Linux/Mac:**
```bash
./build/xor_example
./build/regression_example
```

## Creating Your First Neural Network

### 1. Include the library

```cpp
#include "neural_network.hpp"
using namespace nn;
```

### 2. Prepare your data

```cpp
// Create input tensor (batch_size=4, features=2)
Tensor X({0.0, 0.0,   // Sample 1
          0.0, 1.0,   // Sample 2
          1.0, 0.0,   // Sample 3
          1.0, 1.0},  // Sample 4
         {4, 2}, false);

// Create target tensor
Tensor Y({0.0, 1.0, 1.0, 0.0}, {4, 1}, false);
```

### 3. Build your model

```cpp
auto model = std::make_shared<Sequential>();

// Add layers
model->add(std::make_shared<Linear>(2, 8));   // Input: 2, Hidden: 8
model->add(std::make_shared<ReLU>());         // Activation
model->add(std::make_shared<Linear>(8, 1));   // Output: 1
model->add(std::make_shared<Sigmoid>());      // Output activation
```

### 4. Set up optimizer

```cpp
// Get all model parameters
auto params = model->parameters();

// Create optimizer (Adam with learning rate 0.01)
Adam optimizer(params, 0.01);

// Or use SGD with momentum
// SGD optimizer(params, 0.01, 0.9);
```

### 5. Training loop

```cpp
int num_epochs = 1000;

for (int epoch = 0; epoch < num_epochs; ++epoch) {
    // Forward pass
    Tensor predictions = model->forward(X);
    
    // Compute loss
    Tensor loss = losses::binary_cross_entropy(predictions, Y);
    
    // Backward pass would go here (when fully implemented)
    // loss.backward();
    
    // Update parameters
    optimizer.step();
    
    // Zero gradients for next iteration
    optimizer.zero_grad();
    
    if (epoch % 100 == 0) {
        std::cout << "Epoch " << epoch << ", Loss: " << loss[0] << std::endl;
    }
}
```

### 6. Evaluation

```cpp
// Switch to evaluation mode
model->eval();

// Make predictions
Tensor test_input({0.0, 1.0}, {1, 2}, false);
Tensor prediction = model->forward(test_input);

std::cout << "Prediction: " << prediction[0] << std::endl;
```

## Common Patterns

### Classification

```cpp
auto model = std::make_shared<Sequential>();
model->add(std::make_shared<Linear>(input_size, 128));
model->add(std::make_shared<ReLU>());
model->add(std::make_shared<Dropout>(0.5));
model->add(std::make_shared<Linear>(128, 64));
model->add(std::make_shared<ReLU>());
model->add(std::make_shared<Linear>(64, num_classes));
model->add(std::make_shared<Sigmoid>()); // or Softmax for multi-class

// Use binary_cross_entropy or cross_entropy_loss
Tensor loss = losses::cross_entropy_loss(predictions, targets);
```

### Regression

```cpp
auto model = std::make_shared<Sequential>();
model->add(std::make_shared<Linear>(input_size, 64));
model->add(std::make_shared<ReLU>());
model->add(std::make_shared<Linear>(64, 32));
model->add(std::make_shared<ReLU>());
model->add(std::make_shared<Linear>(32, output_size));
// No activation for regression

// Use MSE or MAE loss
Tensor loss = losses::mse_loss(predictions, targets);
```

### Using Different Optimizers

```cpp
// SGD with momentum
SGD optimizer(params, 0.01, 0.9);

// Adam (recommended for most cases)
Adam optimizer(params, 0.001);

// SGD with weight decay (L2 regularization)
SGD optimizer(params, 0.01, 0.9, 0.0001);
```

### Adjusting Learning Rate

```cpp
// Create optimizer
Adam optimizer(params, 0.01);

// Later, adjust learning rate
if (epoch == 500) {
    optimizer.set_learning_rate(0.001);
}
```

## Tensor Operations

### Creating Tensors

```cpp
// From data
Tensor t1({1.0, 2.0, 3.0, 4.0}, {2, 2});

// Zeros
Tensor zeros = Tensor::zeros({3, 3});

// Ones
Tensor ones = Tensor::ones({2, 4});

// Random normal distribution
Tensor randn = Tensor::randn({5, 5});

// Random uniform distribution
Tensor uniform = Tensor::uniform({3, 3}, -1.0, 1.0);
```

### Basic Operations

```cpp
Tensor a = Tensor::ones({2, 3});
Tensor b = Tensor::ones({2, 3}) * 2.0;

// Element-wise operations
Tensor c = a + b;        // Addition
Tensor d = a - b;        // Subtraction
Tensor e = a * b;        // Element-wise multiplication
Tensor f = a / b;        // Element-wise division

// Scalar operations
Tensor g = a * 2.0;      // Multiply by scalar
Tensor h = a + 1.0;      // Add scalar
```

### Matrix Operations

```cpp
Tensor A({1.0, 2.0, 3.0, 4.0}, {2, 2});
Tensor B({5.0, 6.0, 7.0, 8.0}, {2, 2});

// Matrix multiplication
Tensor C = A.matmul(B);

// Transpose
Tensor At = A.transpose();
```

### Shape Manipulation

```cpp
Tensor t({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3});

// Reshape
Tensor r = t.reshape({3, 2});

// Flatten
Tensor f = t.flatten();  // Shape: {6}

// Slice (axis 0)
Tensor s = t.slice(0, 1);  // First row
```

### Reductions

```cpp
Tensor t = Tensor::randn({10, 10});

// Sum all elements
Tensor sum = t.sum();

// Mean
Tensor mean = t.mean();

// Max and min values
double max_val = t.max();
double min_val = t.min();
```

## Troubleshooting

### Common Errors

1. **Shape mismatch errors**: Ensure your tensor dimensions are compatible
   - For matrix multiplication: (m, k) @ (k, n) = (m, n)
   - For element-wise ops: shapes must match exactly

2. **CMake not found**: Install CMake and add it to your PATH

3. **Compiler errors**: Make sure you have a C++17 compatible compiler

### Getting Help

- Check the full `README.md` for detailed API reference
- Look at example code in `examples/` directory
- Review header files in `include/` for available methods

## Next Steps

- Explore the `examples/` directory for more complete examples
- Read the API reference in `README.md`
- Try building your own networks for different problems
- Experiment with different architectures and hyperparameters

Happy coding! 🚀
