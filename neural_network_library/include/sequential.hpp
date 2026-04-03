#pragma once

/**
 * @file sequential.hpp
 * @brief Sequential module container for chaining layers in order.
 */
#include "module.hpp"
#include <vector>
#include <memory>

namespace nn {

/// Ordered container that applies child modules one after another.
class Sequential : public Module {
public:
    Sequential() = default;
    
    /// Appends a module to the end of the execution pipeline.
    void add(std::shared_ptr<Module> module);
    
    /// Runs the input through every stored module in insertion order.
    Tensor forward(const Tensor& input) override;
    
    /// Collects parameters from every child module.
    std::vector<Tensor*> parameters() override;
    
    /// Switches this container and all child modules into training mode.
    void train();

    /// Switches this container and all child modules into evaluation mode.
    void eval();
    
    /// Returns the number of stored modules.
    size_t size() const { return modules_.size(); }

    /// Returns mutable access to a child module by index.
    std::shared_ptr<Module> operator[](size_t idx) { return modules_[idx]; }

    /// Returns read-only access to a child module by index.
    const std::shared_ptr<Module> operator[](size_t idx) const { return modules_[idx]; }
    
private:
    // Ordered set of layers that define the forward pass.
    std::vector<std::shared_ptr<Module>> modules_;
};

} // namespace nn
