#pragma once

#include "tensor.hpp"
#include <vector>
#include <memory>
#include <unordered_map>

namespace nn {

// Base class for all optimizers
class Optimizer {
public:
    virtual ~Optimizer() = default;
    
    // Perform a single optimization step
    virtual void step() = 0;
    
    // Zero all gradients of parameters
    void zero_grad();
    
protected:
    std::vector<Tensor*> parameters_;
};

// Stochastic Gradient Descent optimizer
class SGD : public Optimizer {
public:
    SGD(const std::vector<Tensor*>& parameters, double learning_rate, 
        double momentum = 0.0, double weight_decay = 0.0);
    
    void step() override;
    
    void set_learning_rate(double lr) { learning_rate_ = lr; }
    double get_learning_rate() const { return learning_rate_; }
    
private:
    double learning_rate_;
    double momentum_;
    double weight_decay_;
    
    // Velocity for momentum
    std::unordered_map<Tensor*, Tensor> velocities_;
};

// Adam optimizer (Adaptive Moment Estimation)
class Adam : public Optimizer {
public:
    Adam(const std::vector<Tensor*>& parameters, double learning_rate = 0.001,
         double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8,
         double weight_decay = 0.0);
    
    void step() override;
    
    void set_learning_rate(double lr) { learning_rate_ = lr; }
    double get_learning_rate() const { return learning_rate_; }
    
private:
    double learning_rate_;
    double beta1_;
    double beta2_;
    double epsilon_;
    double weight_decay_;
    
    size_t t_; // time step
    
    // First moment estimate (mean)
    std::unordered_map<Tensor*, Tensor> m_;
    
    // Second moment estimate (variance)
    std::unordered_map<Tensor*, Tensor> v_;
};

} // namespace nn
