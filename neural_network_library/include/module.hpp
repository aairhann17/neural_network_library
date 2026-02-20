#pragma once

// module.hpp
// Layer abstractions used to build trainable neural networks.
//
// Key pieces:
// - Module: base interface (forward, parameters, mode switching)
// - Linear: fully connected layer with trainable weight/bias
// - ReLU/Sigmoid/Tanh/Dropout: common activation and regularization layers
#include "tensor.hpp"
#include <vector>
#include <memory>
#include <string>

namespace nn {

// Base class for all neural network modules/layers
class Module {
public:
    virtual ~Module() = default;
    
    // Forward pass - must be implemented by derived classes
    virtual Tensor forward(const Tensor& input) = 0;
    
    // Get all parameters (weights, biases, etc.)
    virtual std::vector<Tensor*> parameters() = 0;
    
    // Zero all gradients
    void zero_grad();
    
    // Training/evaluation mode
    void train() { training_ = true; }
    void eval() { training_ = false; }
    bool is_training() const { return training_; }
    
protected:
    bool training_ = true;
};

// Linear (Fully Connected) Layer: y = xW^T + b
class Linear : public Module {
public:
    Linear(size_t in_features, size_t out_features, bool use_bias = true);
    
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor*> parameters() override;
    
    Tensor& weight() { return weight_; }
    Tensor& bias() { return bias_; }
    const Tensor& weight() const { return weight_; }
    const Tensor& bias() const { return bias_; }
    
private:
    size_t in_features_;
    size_t out_features_;
    bool use_bias_;
    
    Tensor weight_;
    Tensor bias_;
    
    void initialize_parameters();
};

// ReLU Activation Layer
class ReLU : public Module {
public:
    ReLU() = default;
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor*> parameters() override { return {}; }
};

// Sigmoid Activation Layer
class Sigmoid : public Module {
public:
    Sigmoid() = default;
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor*> parameters() override { return {}; }
};

// Tanh Activation Layer
class Tanh : public Module {
public:
    Tanh() = default;
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor*> parameters() override { return {}; }
};

// Dropout Layer
class Dropout : public Module {
public:
    explicit Dropout(double p = 0.5);
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor*> parameters() override { return {}; }
    
private:
    double p_; // dropout probability
};

} // namespace nn
