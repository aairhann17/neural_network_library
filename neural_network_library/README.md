# Neural Network Library in C++

A modern C++ neural network library with automatic differentiation support, built from scratch.

## Features

- **Tensor Operations**: Multi-dimensional array support with various operations
- **Automatic Differentiation**: Gradient computation for backpropagation
- **Neural Network Layers**:
  - Linear (Fully Connected) Layer
  - Activation Functions (ReLU, Sigmoid, Tanh, Leaky ReLU, ELU)
  - Dropout
- **Loss Functions**:
  - Mean Squared Error (MSE)
  - Binary Cross Entropy
  - Cross Entropy Loss
  - Mean Absolute Error (MAE)
- **Optimizers**:
  - Stochastic Gradient Descent (SGD) with momentum
  - Adam Optimizer
- **Sequential Model**: Easy model construction

## Project Structure

```
neural_network_library/
├── include/              # Header files
│   ├── tensor.hpp
│   ├── module.hpp
│   ├── sequential.hpp
│   ├── activations.hpp
│   ├── losses.hpp
│   ├── optimizer.hpp
│   └── neural_network.hpp
├── src/                  # Implementation files
│   ├── tensor.cpp
│   ├── module.cpp
│   ├── sequential.cpp
│   ├── activations.cpp
│   ├── losses.cpp
│   └── optimizer.cpp
├── examples/             # Example programs
│   ├── xor_example.cpp
│   └── regression_example.cpp
├── build/                # Build directory
└── CMakeLists.txt
```

## Building the Project

### Prerequisites
- CMake 3.15 or higher
- C++17 compatible compiler (GCC, Clang, MSVC)
- OpenMP (optional, for parallelization)

### Build Instructions

```bash
# Create and navigate to build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build the project
cmake --build .

# On Windows with MSVC, you might use:
cmake --build . --config Release
```

## Usage Examples

### Basic Tensor Operations

```cpp
#include "neural_network.hpp"

using namespace nn;

int main() {
    // Create tensors
    Tensor a = Tensor::ones({2, 3});
    Tensor b = Tensor::randn({2, 3});
    
    // Basic operations
    Tensor c = a + b;
    Tensor d = a * b;  // Element-wise multiplication
    
    // Matrix multiplication
    Tensor m1({1.0, 2.0, 3.0, 4.0}, {2, 2});
    Tensor m2({5.0, 6.0, 7.0, 8.0}, {2, 2});
    Tensor result = m1.matmul(m2);
    
    result.print();
    return 0;
}
```

### Building a Neural Network

```cpp
#include "neural_network.hpp"

using namespace nn;

int main() {
    // Create a neural network
    auto model = std::make_shared<Sequential>();
    model->add(std::make_shared<Linear>(10, 64));
    model->add(std::make_shared<ReLU>());
    model->add(std::make_shared<Dropout>(0.5));
    model->add(std::make_shared<Linear>(64, 32));
    model->add(std::make_shared<ReLU>());
    model->add(std::make_shared<Linear>(32, 1));
    model->add(std::make_shared<Sigmoid>());
    
    // Create optimizer
    Adam optimizer(model->parameters(), 0.001);
    
    // Training loop
    for (int epoch = 0; epoch < 100; ++epoch) {
        // Forward pass
        Tensor output = model->forward(input);
        
        // Compute loss
        Tensor loss = losses::binary_cross_entropy(output, target);
        
        // Backward pass (to be fully implemented with autograd)
        // loss.backward();
        
        // Update parameters
        optimizer.step();
        optimizer.zero_grad();
    }
    
    return 0;
}
```

### Running Examples

After building, you can run the example programs:

```bash
# XOR problem example
./xor_example

# Linear regression example
./regression_example
```

## API Reference

### Tensor Class

```cpp
// Construction
Tensor(shape, requires_grad=false)
Tensor(data, shape, requires_grad=false)

// Static factory methods
Tensor::zeros(shape)
Tensor::ones(shape)
Tensor::randn(shape)
Tensor::uniform(shape, low, high)

// Operations
tensor.matmul(other)
tensor.transpose()
tensor.reshape(new_shape)
tensor.sum(), tensor.mean()

// Autograd
tensor.backward()
tensor.zero_grad()
tensor.grad()
```

### Neural Network Layers

```cpp
Linear(in_features, out_features, use_bias=true)
ReLU()
Sigmoid()
Tanh()
Dropout(p=0.5)
```

### Optimizers

```cpp
SGD(parameters, learning_rate, momentum=0.0, weight_decay=0.0)
Adam(parameters, learning_rate=0.001, beta1=0.9, beta2=0.999)
```

### Loss Functions

```cpp
losses::mse_loss(predictions, targets)
losses::binary_cross_entropy(predictions, targets)
losses::cross_entropy_loss(logits, targets)
losses::mae_loss(predictions, targets)
```

## Roadmap

- [x] Basic tensor operations
- [x] Activation functions
- [x] Loss functions
- [x] Optimizers (SGD, Adam)
- [x] Sequential model
- [ ] Full automatic differentiation
- [ ] Convolutional layers
- [ ] Recurrent layers (LSTM, GRU)
- [ ] Batch normalization
- [ ] Data loaders and datasets
- [ ] Model serialization
- [ ] GPU support

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is open source and available under the MIT License.
