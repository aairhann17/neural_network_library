#include "activations.hpp"

// activations.cpp
// Element-wise and normalized activation implementations.
//
// Notes:
// - Functions are stateless and return a new Tensor.
// - Softmax uses max-subtraction for numerical stability.

#include <algorithm>
#include <cmath>
#include <numeric>

namespace nn {
namespace activations {

Tensor relu(const Tensor& input) {
    Tensor output(input.shape(), input.requires_grad());
    auto mask = std::make_shared<std::vector<double>>(input.size(), 0.0);
    
    for (size_t i = 0; i < input.size(); ++i) {
        if (input[i] > 0.0) {
            output[i] = input[i];
            (*mask)[i] = 1.0;
        } else {
            output[i] = 0.0;
        }
    }

    if (output.requires_grad()) {
        auto input_ptr = Tensor::alias(input);
        auto out_grad = output.grad_ptr();
        output.set_autograd(
            [input_ptr, out_grad, mask]() {
                if (!out_grad || !input_ptr->requires_grad()) {
                    return;
                }
                input_ptr->ensure_grad();
                for (size_t i = 0; i < input_ptr->size(); ++i) {
                    input_ptr->grad()[i] += (*out_grad)[i] * (*mask)[i];
                }
            },
            {input_ptr}
        );
    }
    
    return output;
}

Tensor sigmoid(const Tensor& input) {
    Tensor output(input.shape(), input.requires_grad());
    auto sigmoid_values = std::make_shared<std::vector<double>>(input.size(), 0.0);
    
    for (size_t i = 0; i < input.size(); ++i) {
        const double s = 1.0 / (1.0 + std::exp(-input[i]));
        output[i] = s;
        (*sigmoid_values)[i] = s;
    }

    if (output.requires_grad()) {
        auto input_ptr = Tensor::alias(input);
        auto out_grad = output.grad_ptr();
        output.set_autograd(
            [input_ptr, out_grad, sigmoid_values]() {
                if (!out_grad || !input_ptr->requires_grad()) {
                    return;
                }
                input_ptr->ensure_grad();
                for (size_t i = 0; i < input_ptr->size(); ++i) {
                    const double s = (*sigmoid_values)[i];
                    input_ptr->grad()[i] += (*out_grad)[i] * s * (1.0 - s);
                }
            },
            {input_ptr}
        );
    }
    
    return output;
}

Tensor tanh(const Tensor& input) {
    Tensor output(input.shape(), input.requires_grad());
    
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::tanh(input[i]);
    }
    
    return output;
}

Tensor softmax(const Tensor& input, int axis) {
    if (input.shape().size() != 2) {
        throw std::invalid_argument("Softmax currently only supports 2D tensors");
    }
    
    // Assume axis = -1 or 1 (apply softmax across columns for each row)
    size_t rows = input.shape()[0];
    size_t cols = input.shape()[1];
    
    Tensor output(input.shape(), input.requires_grad());
    
    for (size_t i = 0; i < rows; ++i) {
        // Find max for numerical stability
        double max_val = input[i * cols];
        for (size_t j = 1; j < cols; ++j) {
            max_val = std::max(max_val, input[i * cols + j]);
        }
        
        // Compute exp and sum
        double sum_exp = 0.0;
        for (size_t j = 0; j < cols; ++j) {
            double exp_val = std::exp(input[i * cols + j] - max_val);
            output[i * cols + j] = exp_val;
            sum_exp += exp_val;
        }
        
        // Normalize
        for (size_t j = 0; j < cols; ++j) {
            output[i * cols + j] /= sum_exp;
        }
    }
    
    return output;
}

Tensor leaky_relu(const Tensor& input, double alpha) {
    Tensor output(input.shape(), input.requires_grad());
    
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] > 0.0 ? input[i] : alpha * input[i];
    }
    
    return output;
}

Tensor elu(const Tensor& input, double alpha) {
    Tensor output(input.shape(), input.requires_grad());
    
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] > 0.0 ? input[i] : alpha * (std::exp(input[i]) - 1.0);
    }
    
    return output;
}

} // namespace activations
} // namespace nn
