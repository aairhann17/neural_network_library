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

/** @ingroup module_api */
/// Base interface for all trainable or stateless neural-network layers.
///
/// A Module consumes an input tensor in forward(), optionally exposes trainable
/// parameters(), and keeps track of whether the layer should behave in training
/// mode or evaluation mode.
class Module {
public:
    /** @brief Virtual destructor for polymorphic layer hierarchy. */
    virtual ~Module() = default;
    
    /**
     * @brief Runs layer forward pass.
     * @param input Input tensor.
     * @return Output tensor produced by this layer.
     */
    virtual Tensor forward(const Tensor& input) = 0;
    
    /**
     * @brief Returns trainable parameters owned by this layer.
     * @return Vector of non-owning pointers to parameter tensors.
     */
    virtual std::vector<Tensor*> parameters() = 0;
    
    /** @brief Zeros gradients of all parameters returned by parameters(). */
    void zero_grad();
    
    /** @brief Switches layer behavior to training mode. */
    void train() { training_ = true; }

    /** @brief Switches layer behavior to evaluation mode. */
    void eval() { training_ = false; }

    /**
     * @brief Indicates whether the layer is in training mode.
     * @return True when layer behaves in training mode.
     */
    bool is_training() const { return training_; }
    
protected:
    // Shared mode flag used by derived layers such as Dropout.
    bool training_ = true;
};

/** @ingroup module_api */
/// Fully connected affine layer implementing $y = xW^T + b$.
class Linear : public Module {
public:
    /**
     * @brief Constructs fully connected layer.
     * @param in_features Number of input features.
     * @param out_features Number of output features.
     * @param use_bias Whether to include additive bias term.
     */
    Linear(size_t in_features, size_t out_features, bool use_bias = true);
    
    /**
     * @brief Applies affine transform to batched 2D input.
     * @param input Input tensor of shape (batch_size, in_features).
     * @return Output tensor of shape (batch_size, out_features).
     * @throws std::invalid_argument If input rank is not 2 or feature size mismatches in_features.
     */
    Tensor forward(const Tensor& input) override;

    /**
     * @brief Returns trainable tensors of this layer.
     * @return Vector containing weight and, when enabled, bias.
     */
    std::vector<Tensor*> parameters() override;
    
    /**
     * @brief Returns mutable weight matrix.
     * @return Weight tensor reference.
     */
    Tensor& weight() { return weight_; }

    /**
     * @brief Returns mutable bias vector.
     * @return Bias tensor reference.
     */
    Tensor& bias() { return bias_; }

    /**
     * @brief Returns read-only weight matrix.
     * @return Const weight tensor reference.
     */
    const Tensor& weight() const { return weight_; }

    /**
     * @brief Returns read-only bias vector.
     * @return Const bias tensor reference.
     */
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
    
    /** @brief Initializes trainable parameters. */
    void initialize_parameters();
};

/** @ingroup module_api */
/// Module wrapper around the ReLU activation function.
class ReLU : public Module {
public:
    /** @brief Constructs ReLU layer. */
    ReLU() = default;

    /**
     * @brief Applies ReLU activation.
     * @param input Input tensor.
     * @return Activated tensor.
     */
    Tensor forward(const Tensor& input) override;

    /** @brief ReLU has no trainable parameters. */
    std::vector<Tensor*> parameters() override { return {}; }
};

/** @ingroup module_api */
/// Module wrapper around the sigmoid activation function.
class Sigmoid : public Module {
public:
    /** @brief Constructs sigmoid layer. */
    Sigmoid() = default;

    /**
     * @brief Applies sigmoid activation.
     * @param input Input tensor.
     * @return Activated tensor.
     */
    Tensor forward(const Tensor& input) override;

    /** @brief Sigmoid has no trainable parameters. */
    std::vector<Tensor*> parameters() override { return {}; }
};

/** @ingroup module_api */
/// Module wrapper around the hyperbolic tangent activation function.
class Tanh : public Module {
public:
    /** @brief Constructs hyperbolic tangent layer. */
    Tanh() = default;

    /**
     * @brief Applies tanh activation.
     * @param input Input tensor.
     * @return Activated tensor.
     */
    Tensor forward(const Tensor& input) override;

    /** @brief Tanh has no trainable parameters. */
    std::vector<Tensor*> parameters() override { return {}; }
};

/** @ingroup module_api */
/// Regularization layer that randomly zeroes activations during training.
class Dropout : public Module {
public:
    /**
     * @brief Constructs dropout layer.
     * @param p Probability of masking each activation.
     * @throws std::invalid_argument If p is outside range [0, 1].
     */
    explicit Dropout(double p = 0.5);

    /**
     * @brief Applies inverted dropout in training mode.
     * @param input Input tensor.
     * @return Input-like tensor with randomly masked elements in training mode.
     */
    Tensor forward(const Tensor& input) override;

    /** @brief Dropout has no trainable parameters. */
    std::vector<Tensor*> parameters() override { return {}; }
    
private:
    // Probability that a value is masked during training.
    double p_;
};

} // namespace nn
