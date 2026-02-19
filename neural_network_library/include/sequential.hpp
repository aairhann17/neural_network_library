#pragma once

#include "module.hpp"
#include <vector>
#include <memory>

namespace nn {

// Sequential container for neural network layers
class Sequential : public Module {
public:
    Sequential() = default;
    
    // Add a layer to the sequential model
    void add(std::shared_ptr<Module> module);
    
    // Forward pass through all layers
    Tensor forward(const Tensor& input) override;
    
    // Get all parameters from all layers
    std::vector<Tensor*> parameters() override;
    
    // Set training/evaluation mode for all layers
    void train();
    void eval();
    
    // Access individual layers
    size_t size() const { return modules_.size(); }
    std::shared_ptr<Module> operator[](size_t idx) { return modules_[idx]; }
    const std::shared_ptr<Module> operator[](size_t idx) const { return modules_[idx]; }
    
private:
    std::vector<std::shared_ptr<Module>> modules_;
};

} // namespace nn
