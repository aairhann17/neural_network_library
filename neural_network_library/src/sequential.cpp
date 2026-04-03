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
    // Preserve insertion order because it defines the model's forward graph.
    modules_.push_back(module);
}

Tensor Sequential::forward(const Tensor& input) {
    if (modules_.empty()) {
        throw std::runtime_error("Sequential model has no layers");
    }

    // Alternate between two temporary buffers to avoid creating a named tensor
    // variable for every layer application in the loop.
    Tensor buffer_a = input;
    Tensor buffer_b = input;
    Tensor* current = &buffer_a;
    Tensor* next = &buffer_b;

    for (auto& module : modules_) {
        // Each layer consumes the current buffer and writes its result to the
        // spare buffer, after which the pointers are swapped.
        *next = module->forward(*current);
        std::swap(current, next);
    }

    return *current;
}

std::vector<Tensor*> Sequential::parameters() {
    std::vector<Tensor*> params;
    
    for (auto& module : modules_) {
        // Flatten child parameter lists into one optimizer-friendly vector.
        auto module_params = module->parameters();
        params.insert(params.end(), module_params.begin(), module_params.end());
    }
    
    return params;
}

void Sequential::train() {
    // Update the container state first, then propagate the mode to children.
    training_ = true;
    for (auto& module : modules_) {
        module->train();
    }
}

void Sequential::eval() {
    // Evaluation mode is especially important for stochastic layers like Dropout.
    training_ = false;
    for (auto& module : modules_) {
        module->eval();
    }
}

} // namespace nn
