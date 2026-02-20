#include "module.hpp"
#include "activations.hpp"
#include <cmath>
#include <random>

namespace nn {
// module.cpp
// Implementations for layer modules (Linear, activations, Dropout)
// and shared Module utilities.
//
// Linear layer summary:
// - Parameters are initialized with Xavier-style random weights.
// - forward() computes input @ weight^T (+ bias when enabled).

// ============ Module Base Class ============

void Module::zero_grad() {
    for (auto* param : parameters()) {
        param->zero_grad();
    }
}

// ============ Linear Layer ============

Linear::Linear(size_t in_features, size_t out_features, bool use_bias)
    : in_features_(in_features),
      out_features_(out_features),
      use_bias_(use_bias),
      weight_({out_features, in_features}, true),
      bias_({out_features}, true) {
    
    initialize_parameters();
}

void Linear::initialize_parameters() {
    // Xavier/Glorot initialization
    double std = std::sqrt(2.0 / static_cast<double>(in_features_ + out_features_));
    
    weight_ = Tensor::randn({out_features_, in_features_}, true);
    weight_ = weight_ * std;
    
    if (use_bias_) {
        bias_ = Tensor::zeros({out_features_}, true);
    }
}

Tensor Linear::forward(const Tensor& input) {
    // input shape: (batch_size, in_features)
    // weight shape: (out_features, in_features)
    // output shape: (batch_size, out_features)
    
    if (input.shape().size() != 2) {
        throw std::invalid_argument("Linear layer expects 2D input (batch_size, in_features)");
    }
    
    if (input.shape()[1] != in_features_) {
        throw std::invalid_argument(
            "Input features mismatch. Expected: " + std::to_string(in_features_) +
            ", got: " + std::to_string(input.shape()[1])
        );
    }
    
    // Matrix multiplication: input @ weight^T
    Tensor output = input.matmul(weight_.transpose());
    
    // Add bias if used
    if (use_bias_) {
        // Broadcast bias across batch dimension
        for (size_t i = 0; i < output.shape()[0]; ++i) {
            for (size_t j = 0; j < output.shape()[1]; ++j) {
                output[i * output.shape()[1] + j] += bias_[j];
            }
        }
    }
    
    return output;
}

std::vector<Tensor*> Linear::parameters() {
    if (use_bias_) {
        return {&weight_, &bias_};
    } else {
        return {&weight_};
    }
}

// ============ Activation Layers ============

Tensor ReLU::forward(const Tensor& input) {
    return activations::relu(input);
}

Tensor Sigmoid::forward(const Tensor& input) {
    return activations::sigmoid(input);
}

Tensor Tanh::forward(const Tensor& input) {
    return activations::tanh(input);
}

// ============ Dropout Layer ============

Dropout::Dropout(double p) : p_(p) {
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("Dropout probability must be between 0 and 1");
    }
}

Tensor Dropout::forward(const Tensor& input) {
    if (!is_training()) {
        return input; // No dropout during evaluation
    }
    
    Tensor output(input.shape(), input.requires_grad());
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(1.0 - p_);
    
    double scale = 1.0 / (1.0 - p_);
    
    for (size_t i = 0; i < input.size(); ++i) {
        if (dist(gen)) {
            output[i] = input[i] * scale;
        } else {
            output[i] = 0.0;
        }
    }
    
    return output;
}

} // namespace nn
