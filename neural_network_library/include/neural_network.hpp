#pragma once

/**
 * @file neural_network.hpp
 * @brief Umbrella header that re-exports the library's public API.
 */

// Core tensor operations
#include "tensor.hpp"

// Neural network modules and layers
#include "module.hpp"
#include "sequential.hpp"

// Activation functions
#include "activations.hpp"

// Loss functions
#include "losses.hpp"

// Optimizers
#include "optimizer.hpp"

// The library intentionally keeps all public types in namespace nn.
namespace nn {
}
