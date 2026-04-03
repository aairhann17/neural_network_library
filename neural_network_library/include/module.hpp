#pragma once

/**
 * @file module.hpp
 * @brief Layer abstractions used to assemble and train neural networks.
 */
#include "tensor.hpp"
#include <vector>
#include <memory>
#include <string>

namespace nn {

/// Base interface for all trainable or stateless neural-network layers.
///
/// A Module consumes an input tensor in forward(), optionally exposes trainable
/// parameters(), and keeps track of whether the layer should behave in training
/// mode or evaluation mode.
class Module {
public:
    virtual ~Module() = default;
    
    /// Runs the layer's forward pass.
    virtual Tensor forward(const Tensor& input) = 0;
    
    /// Returns pointers to all trainable tensors owned by the module.
    virtual std::vector<Tensor*> parameters() = 0;
    
    /// Zeros all parameter gradients owned by the module.
    void zero_grad();
    
    /// Switches the module into training mode.
    void train() { training_ = true; }

    /// Switches the module into evaluation mode.
    void eval() { training_ = false; }

    /// Returns whether the module is currently in training mode.
    bool is_training() const { return training_; }
    
protected:
    // Shared mode flag used by derived layers such as Dropout.
    bool training_ = true;
};

/// Fully connected affine layer implementing $y = xW^T + b$.
class Linear : public Module {
public:
    /// Constructs a linear layer with optional bias.
    Linear(size_t in_features, size_t out_features, bool use_bias = true);
    
    /// Applies the affine transform to a batch of inputs.
    Tensor forward(const Tensor& input) override;

    /// Returns the layer's trainable weight and optional bias tensors.
    std::vector<Tensor*> parameters() override;
    
    /// Returns mutable access to the weight matrix.
    Tensor& weight() { return weight_; }

    /// Returns mutable access to the bias vector.
    Tensor& bias() { return bias_; }

    /// Returns read-only access to the weight matrix.
    const Tensor& weight() const { return weight_; }

    /// Returns read-only access to the bias vector.
    const Tensor& bias() const { return bias_; }
    
private:
    // Number of input features expected per sample.
    size_t in_features_;

    // Number of output features produced per sample.
    size_t out_features_;

    // Whether the layer includes an additive bias term.
    bool use_bias_;
    
    // Trainable weight matrix of shape (out_features, in_features).
    Tensor weight_;

    // Trainable bias vector of shape (out_features).
    Tensor bias_;
    
    /// Initializes learnable parameters with reasonable defaults.
    void initialize_parameters();
};

/// Module wrapper around the ReLU activation function.
class ReLU : public Module {
public:
    ReLU() = default;
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor*> parameters() override { return {}; }
};

/// Module wrapper around the sigmoid activation function.
class Sigmoid : public Module {
public:
    Sigmoid() = default;
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor*> parameters() override { return {}; }
};

/// Module wrapper around the hyperbolic tangent activation function.
class Tanh : public Module {
public:
    Tanh() = default;
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor*> parameters() override { return {}; }
};

/// Regularization layer that randomly zeroes activations during training.
class Dropout : public Module {
public:
    /// Constructs dropout with probability p of dropping an activation.
    explicit Dropout(double p = 0.5);
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor*> parameters() override { return {}; }
    
private:
    // Probability that a value is masked during training.
    double p_;
};

} // namespace nn
