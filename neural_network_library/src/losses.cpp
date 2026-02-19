#include "losses.hpp"
#include "activations.hpp"
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
    Tensor loss(predictions.shape(), predictions.requires_grad());
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        double p = std::max(epsilon, std::min(1.0 - epsilon, predictions[i]));
        double y = targets[i];
        loss[i] = -(y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
    }
    
    return loss.mean();
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
