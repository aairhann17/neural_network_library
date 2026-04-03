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
    /** @brief Constructs an empty sequential model. */
    Sequential() = default;
    
    /**
     * @brief Appends layer to end of pipeline.
     * @param module Shared pointer to layer to append.
     */
    void add(std::shared_ptr<Module> module);
    
    /**
     * @brief Runs forward pass through all child layers in order.
     * @param input Input tensor.
     * @return Output tensor from final layer.
     * @throws std::runtime_error If no layers were added.
     */
    Tensor forward(const Tensor& input) override;
    
    /**
     * @brief Collects trainable parameters from all child layers.
     * @return Flattened list of parameter tensor pointers.
     */
    std::vector<Tensor*> parameters() override;
    
    /** @brief Switches this container and all child modules to training mode. */
    void train();

    /** @brief Switches this container and all child modules to evaluation mode. */
    void eval();
    
    /**
     * @brief Returns number of child modules.
     * @return Layer count in this container.
     */
    size_t size() const { return modules_.size(); }

    /**
     * @brief Returns mutable access to child module by index.
     * @param idx Zero-based layer index.
     * @return Shared pointer to module at idx.
     */
    std::shared_ptr<Module> operator[](size_t idx) { return modules_[idx]; }

    /**
     * @brief Returns read-only access to child module by index.
     * @param idx Zero-based layer index.
     * @return Const shared pointer to module at idx.
     */
    const std::shared_ptr<Module> operator[](size_t idx) const { return modules_[idx]; }
    
private:
    // Ordered set of layers that define the forward pass.
    std::vector<std::shared_ptr<Module>> modules_;
};

} // namespace nn
