#pragma once

#include "tensor.hpp"
#include <cmath>

namespace nn {

// Activation functions
namespace activations {

// ReLU (Rectified Linear Unit): f(x) = max(0, x)
Tensor relu(const Tensor& input);

// Sigmoid: f(x) = 1 / (1 + exp(-x))
Tensor sigmoid(const Tensor& input);

// Tanh: f(x) = tanh(x)
Tensor tanh(const Tensor& input);

// Softmax: f(x_i) = exp(x_i) / sum(exp(x_j))
Tensor softmax(const Tensor& input, int axis = -1);

// Leaky ReLU: f(x) = x if x > 0 else alpha * x
Tensor leaky_relu(const Tensor& input, double alpha = 0.01);

// ELU (Exponential Linear Unit): f(x) = x if x > 0 else alpha * (exp(x) - 1)
Tensor elu(const Tensor& input, double alpha = 1.0);

} // namespace activations

} // namespace nn
