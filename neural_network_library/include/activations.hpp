#pragma once

/**
 * @file activations.hpp
 * @brief Stateless activation helpers for tensor-valued computations.
 */
#include "tensor.hpp"
#include <cmath>

namespace nn {

/** @ingroup activation_api */
/// Stateless activation helpers used by module wrappers and direct tensor code.
namespace activations {

/** @ingroup activation_api */
/**
 * @brief Applies ReLU activation, f(x) = max(0, x).
 * @param input Input tensor.
 * @return Tensor with ReLU applied element-wise.
 */
Tensor relu(const Tensor& input);

/** @ingroup activation_api */
/**
 * @brief Applies sigmoid activation, f(x) = 1 / (1 + exp(-x)).
 * @param input Input tensor.
 * @return Tensor with sigmoid applied element-wise.
 */
Tensor sigmoid(const Tensor& input);

/** @ingroup activation_api */
/**
 * @brief Applies hyperbolic tangent activation.
 * @param input Input tensor.
 * @return Tensor with tanh applied element-wise.
 */
Tensor tanh(const Tensor& input);

/** @ingroup activation_api */
/**
 * @brief Applies numerically stable softmax normalization.
 * @param input Input tensor, currently expected to be 2D.
 * @param axis Axis to normalize along, currently row-wise behavior is implemented.
 * @return Tensor containing normalized probabilities.
 * @throws std::invalid_argument If input is not 2D.
 */
Tensor softmax(const Tensor& input, int axis = -1);

/** @ingroup activation_api */
/**
 * @brief Applies leaky ReLU activation.
 * @param input Input tensor.
 * @param alpha Negative-slope coefficient.
 * @return Tensor with leaky ReLU applied element-wise.
 */
Tensor leaky_relu(const Tensor& input, double alpha = 0.01);

/** @ingroup activation_api */
/**
 * @brief Applies ELU activation.
 * @param input Input tensor.
 * @param alpha ELU scaling coefficient for negative branch.
 * @return Tensor with ELU applied element-wise.
 */
Tensor elu(const Tensor& input, double alpha = 1.0);

} // namespace activations

} // namespace nn
