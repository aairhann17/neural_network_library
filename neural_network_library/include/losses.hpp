#pragma once

#include "tensor.hpp"

namespace nn {
namespace losses {

// Mean Squared Error: MSE = mean((y_pred - y_true)^2)
Tensor mse_loss(const Tensor& predictions, const Tensor& targets);

// Binary Cross Entropy: BCE = -mean(y * log(p) + (1-y) * log(1-p))
Tensor binary_cross_entropy(const Tensor& predictions, const Tensor& targets);

// Cross Entropy Loss (with softmax): CE = -sum(y * log(softmax(x)))
Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets);

// Mean Absolute Error: MAE = mean(|y_pred - y_true|)
Tensor mae_loss(const Tensor& predictions, const Tensor& targets);

} // namespace losses
} // namespace nn
