#include "losses.hpp"
#include "activations.hpp"

// losses.cpp
// Loss function implementations used during training.
//
// Design notes:
// - Inputs are validated for shape compatibility.
// - Losses return Tensor values so they can be part of autograd graphs.

#include <cmath>
#include <algorithm>

namespace nn {
namespace losses {

Tensor mse_loss(const Tensor& predictions, const Tensor& targets) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    
    Tensor diff = predictions - targets;
    Tensor squared = diff * diff;
    return squared.mean();
}

Tensor binary_cross_entropy(const Tensor& predictions, const Tensor& targets) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    
    const double epsilon = 1e-7; // for numerical stability
    auto dloss_dpred = std::make_shared<std::vector<double>>(predictions.size(), 0.0);
    double total_loss = 0.0;
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        double p = std::max(epsilon, std::min(1.0 - epsilon, predictions[i]));
        double y = targets[i];
        const double sample_loss = -(y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
        total_loss += sample_loss;
        (*dloss_dpred)[i] = (-((y / p) - ((1.0 - y) / (1.0 - p)))) /
                           static_cast<double>(predictions.size());
    }

    const double mean_loss = total_loss / static_cast<double>(predictions.size());
    Tensor result({mean_loss}, {1}, predictions.requires_grad());

    if (result.requires_grad()) {
        auto pred_ptr = Tensor::alias(predictions);
        auto out_grad = result.grad_ptr();
        result.set_autograd(
            [pred_ptr, out_grad, dloss_dpred]() {
                if (!out_grad || !pred_ptr->requires_grad()) {
                    return;
                }
                pred_ptr->ensure_grad();
                for (size_t i = 0; i < pred_ptr->size(); ++i) {
                    pred_ptr->grad()[i] += (*out_grad)[0] * (*dloss_dpred)[i];
                }
            },
            {pred_ptr}
        );
    }
    
    return result;
}

Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets) {
    if (logits.shape().size() != 2 || targets.shape().size() != 2) {
        throw std::invalid_argument("Cross entropy expects 2D tensors (batch_size x num_classes)");
    }
    
    if (logits.shape()[0] != targets.shape()[0]) {
        throw std::invalid_argument("Batch sizes must match");
    }
    
    // Apply softmax to logits
    Tensor probs = activations::softmax(logits);
    
    const double epsilon = 1e-7;
    size_t batch_size = logits.shape()[0];
    size_t num_classes = logits.shape()[1];
    
    double total_loss = 0.0;
    
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < num_classes; ++j) {
            size_t idx = i * num_classes + j;
            if (targets[idx] > 0.0) { // only compute for non-zero targets
                double p = std::max(epsilon, probs[idx]);
                total_loss -= targets[idx] * std::log(p);
            }
        }
    }
    
    total_loss /= static_cast<double>(batch_size);
    
    return Tensor({total_loss}, {1}, logits.requires_grad());
}

Tensor mae_loss(const Tensor& predictions, const Tensor& targets) {
    if (predictions.shape() != targets.shape()) {
        throw std::invalid_argument("Predictions and targets must have the same shape");
    }
    
    Tensor diff = predictions - targets;
    Tensor abs_diff(diff.shape(), diff.requires_grad());
    
    for (size_t i = 0; i < diff.size(); ++i) {
        abs_diff[i] = std::abs(diff[i]);
    }
    
    return abs_diff.mean();
}

} // namespace losses
} // namespace nn
