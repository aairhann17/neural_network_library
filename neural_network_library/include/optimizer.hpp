#pragma once

/**
 * @file optimizer.hpp
 * @brief Parameter update rules used during model training.
 */
#include "tensor.hpp"
#include <vector>
#include <memory>
#include <unordered_map>

namespace nn {

/// Base class for parameter update rules.
class Optimizer {
public:
    virtual ~Optimizer() = default;
    
    /// Applies one optimization step to every managed parameter.
    virtual void step() = 0;
    
    /// Zeros gradients for all managed parameters.
    void zero_grad();
    
protected:
    // Non-owning pointers to trainable tensors updated by the optimizer.
    std::vector<Tensor*> parameters_;
};

/// Stochastic gradient descent with optional momentum and weight decay.
class SGD : public Optimizer {
public:
    /// Constructs an SGD optimizer for the provided parameter list.
    SGD(const std::vector<Tensor*>& parameters, double learning_rate, 
        double momentum = 0.0, double weight_decay = 0.0);
    
    void step() override;
    
    /// Updates the optimizer learning rate.
    void set_learning_rate(double lr) { learning_rate_ = lr; }

    /// Returns the current learning rate.
    double get_learning_rate() const { return learning_rate_; }
    
private:
    // Step size used during parameter updates.
    double learning_rate_;

    // Momentum coefficient used for velocity accumulation.
    double momentum_;

    // L2 penalty coefficient.
    double weight_decay_;
    
    // Per-parameter velocity buffers used when momentum is enabled.
    std::unordered_map<Tensor*, Tensor> velocities_;
};

/// Adam optimizer with bias-corrected first and second moments.
class Adam : public Optimizer {
public:
    /// Constructs an Adam optimizer for the provided parameter list.
    Adam(const std::vector<Tensor*>& parameters, double learning_rate = 0.001,
         double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8,
         double weight_decay = 0.0);
    
    void step() override;
    
    /// Updates the optimizer learning rate.
    void set_learning_rate(double lr) { learning_rate_ = lr; }

    /// Returns the current learning rate.
    double get_learning_rate() const { return learning_rate_; }
    
private:
    // Step size used during parameter updates.
    double learning_rate_;

    // Exponential decay rate for the first moment estimate.
    double beta1_;

    // Exponential decay rate for the second moment estimate.
    double beta2_;

    // Small constant that avoids division by zero.
    double epsilon_;

    // L2 penalty coefficient.
    double weight_decay_;
    
    // Optimizer time step used for bias correction.
    size_t t_;
    
    // Per-parameter first moment buffers.
    std::unordered_map<Tensor*, Tensor> m_;
    
    // Per-parameter second moment buffers.
    std::unordered_map<Tensor*, Tensor> v_;
};

} // namespace nn
