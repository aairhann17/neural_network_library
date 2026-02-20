#include "sequential.hpp"

// sequential.cpp
// Sequential model container implementation.
//
// Responsibilities:
// - Maintain ordered module list
// - Forward input through modules in order
// - Aggregate parameters across all child modules

namespace nn {

void Sequential::add(std::shared_ptr<Module> module) {
    modules_.push_back(module);
}

Tensor Sequential::forward(const Tensor& input) {
    if (modules_.empty()) {
        throw std::runtime_error("Sequential model has no layers");
    }
    
    Tensor output = input;
    
    for (auto& module : modules_) {
        output = module->forward(output);
    }
    
    return output;
}

std::vector<Tensor*> Sequential::parameters() {
    std::vector<Tensor*> params;
    
    for (auto& module : modules_) {
        auto module_params = module->parameters();
        params.insert(params.end(), module_params.begin(), module_params.end());
    }
    
    return params;
}

void Sequential::train() {
    training_ = true;
    for (auto& module : modules_) {
        module->train();
    }
}

void Sequential::eval() {
    training_ = false;
    for (auto& module : modules_) {
        module->eval();
    }
}

} // namespace nn
