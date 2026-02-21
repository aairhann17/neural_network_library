#include "optimizer.hpp"

// optimizer.cpp
// Implementations of parameter update rules (SGD and Adam).
//
// Training step pattern:
// - Read each parameter's gradient
// - Optionally apply weight decay
// - Update parameter values in-place

#include <cmath>
#include <algorithm>

namespace nn {

// ============ Optimizer Base Class ============

void Optimizer::zero_grad() {
    for (auto* param : parameters_) {
        if (param->has_grad()) {
            param->zero_grad();
        }
    }
}

// ============ SGD Optimizer ============

SGD::SGD(const std::vector<Tensor*>& parameters, double learning_rate,
         double momentum, double weight_decay)
    : learning_rate_(learning_rate),
      momentum_(momentum),
      weight_decay_(weight_decay) {
    
    parameters_ = parameters;
    
    // Initialize velocity tensors for momentum
    if (momentum_ > 0.0) {
        for (auto* param : parameters_) {
            velocities_.emplace(param, Tensor::zeros(param->shape()));
        }
    }
}

void SGD::step() {
    for (auto* param : parameters_) {
        if (!param->has_grad()) {
            continue;
        }
        
        const Tensor& grad = param->grad();
        
        // Apply weight decay (L2 regularization)
        Tensor effective_grad = grad;
        if (weight_decay_ > 0.0) {
            // grad += weight_decay * param
            for (size_t i = 0; i < param->size(); ++i) {
                effective_grad[i] += weight_decay_ * (*param)[i];
            }
        }
        
        if (momentum_ > 0.0) {
            // Velocity update: v = momentum * v + grad
            auto velocity_it = velocities_.find(param);
            if (velocity_it == velocities_.end()) {
                velocity_it = velocities_.emplace(param, Tensor::zeros(param->shape())).first;
            }
            Tensor& velocity = velocity_it->second;
            for (size_t i = 0; i < velocity.size(); ++i) {
                velocity[i] = momentum_ * velocity[i] + effective_grad[i];
            }
            
            // Parameter update: param -= learning_rate * velocity
            for (size_t i = 0; i < param->size(); ++i) {
                (*param)[i] -= learning_rate_ * velocity[i];
            }
        } else {
            // Standard SGD: param -= learning_rate * grad
            for (size_t i = 0; i < param->size(); ++i) {
                (*param)[i] -= learning_rate_ * effective_grad[i];
            }
        }
    }
}

// ============ Adam Optimizer ============

Adam::Adam(const std::vector<Tensor*>& parameters, double learning_rate,
           double beta1, double beta2, double epsilon, double weight_decay)
    : learning_rate_(learning_rate),
      beta1_(beta1),
      beta2_(beta2),
      epsilon_(epsilon),
      weight_decay_(weight_decay),
      t_(0) {
    
    parameters_ = parameters;
    
    // Initialize first and second moment estimates
    for (auto* param : parameters_) {
        m_.emplace(param, Tensor::zeros(param->shape()));
        v_.emplace(param, Tensor::zeros(param->shape()));
    }
}

void Adam::step() {
    t_++; // Increment time step
    
    for (auto* param : parameters_) {
        if (!param->has_grad()) {
            continue;
        }
        
        const Tensor& grad = param->grad();
        
        // Apply weight decay (L2 regularization)
        Tensor effective_grad = grad;
        if (weight_decay_ > 0.0) {
            for (size_t i = 0; i < param->size(); ++i) {
                effective_grad[i] += weight_decay_ * (*param)[i];
            }
        }
        
        auto m_it = m_.find(param);
        if (m_it == m_.end()) {
            m_it = m_.emplace(param, Tensor::zeros(param->shape())).first;
        }
        auto v_it = v_.find(param);
        if (v_it == v_.end()) {
            v_it = v_.emplace(param, Tensor::zeros(param->shape())).first;
        }

        Tensor& m = m_it->second; // first moment
        Tensor& v = v_it->second; // second moment
        
        // Update biased first and second moment estimates
        for (size_t i = 0; i < param->size(); ++i) {
            // m = beta1 * m + (1 - beta1) * grad
            m[i] = beta1_ * m[i] + (1.0 - beta1_) * effective_grad[i];
            
            // v = beta2 * v + (1 - beta2) * grad^2
            v[i] = beta2_ * v[i] + (1.0 - beta2_) * effective_grad[i] * effective_grad[i];
            
            // Compute bias-corrected estimates
            double m_hat = m[i] / (1.0 - std::pow(beta1_, t_));
            double v_hat = v[i] / (1.0 - std::pow(beta2_, t_));
            
            // Update parameters
            (*param)[i] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }
}

} // namespace nn
