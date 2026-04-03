#pragma once

/**
 * @file losses.hpp
 * @brief Loss functions for supervised learning workflows.
 */
#include "tensor.hpp"

namespace nn {
namespace losses {

/** @ingroup loss_api */
/**
 * @brief Computes mean squared error.
 * @param predictions Model predictions.
 * @param targets Ground-truth values.
 * @return Scalar tensor of shape {1} containing MSE value.
 * @throws std::invalid_argument If predictions and targets shapes differ.
 */
Tensor mse_loss(const Tensor& predictions, const Tensor& targets);

/** @ingroup loss_api */
/**
 * @brief Computes binary cross-entropy loss.
 * @param predictions Predicted probabilities.
 * @param targets Binary ground-truth values.
 * @return Scalar tensor of shape {1} containing BCE value.
 * @throws std::invalid_argument If predictions and targets shapes differ.
 */
Tensor binary_cross_entropy(const Tensor& predictions, const Tensor& targets);

/** @ingroup loss_api */
/**
 * @brief Computes multiclass cross-entropy from logits and targets.
 * @param logits Model logits, typically shape (batch_size, num_classes).
 * @param targets One-hot or class-probability targets with same batch and class dimensions.
 * @return Scalar tensor of shape {1} containing average cross-entropy.
 * @throws std::invalid_argument If logits/targets ranks or batch dimensions are invalid.
 */
Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets);

/** @ingroup loss_api */
/**
 * @brief Computes mean absolute error.
 * @param predictions Model predictions.
 * @param targets Ground-truth values.
 * @return Scalar tensor of shape {1} containing MAE value.
 * @throws std::invalid_argument If predictions and targets shapes differ.
 */
Tensor mae_loss(const Tensor& predictions, const Tensor& targets);

} // namespace losses
} // namespace nn
