#pragma once

/**
 * @file activations.hpp
 * @brief Stateless activation helpers for tensor-valued computations.
 */
#include "tensor.hpp"
#include <cmath>

namespace nn {

/// Stateless activation helpers used by module wrappers and direct tensor code.
namespace activations {

/// ReLU (Rectified Linear Unit): $f(x) = \max(0, x)$.
Tensor relu(const Tensor& input);

/// Sigmoid: $f(x) = \frac{1}{1 + e^{-x}}$.
Tensor sigmoid(const Tensor& input);

/// Hyperbolic tangent: $f(x) = \tanh(x)$.
Tensor tanh(const Tensor& input);

/// Softmax along the requested axis with numerical stabilization.
Tensor softmax(const Tensor& input, int axis = -1);

/// Leaky ReLU: $f(x) = x$ for positive inputs and $\alpha x$ otherwise.
Tensor leaky_relu(const Tensor& input, double alpha = 0.01);

/// ELU: $f(x) = x$ for positive inputs and $\alpha (e^x - 1)$ otherwise.
Tensor elu(const Tensor& input, double alpha = 1.0);

} // namespace activations

} // namespace nn
