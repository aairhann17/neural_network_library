#pragma once

/**
 * @file losses.hpp
 * @brief Loss functions for supervised learning workflows.
 */
#include "tensor.hpp"

namespace nn {
namespace losses {

/// Mean squared error averaged across all prediction elements.
Tensor mse_loss(const Tensor& predictions, const Tensor& targets);

/// Binary cross-entropy loss for probabilities in the range $[0, 1]$.
Tensor binary_cross_entropy(const Tensor& predictions, const Tensor& targets);

/// Multiclass cross-entropy loss computed from logits and one-hot targets.
Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets);

/// Mean absolute error averaged across all prediction elements.
Tensor mae_loss(const Tensor& predictions, const Tensor& targets);

} // namespace losses
} // namespace nn
