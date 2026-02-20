#pragma once

// neural_network.hpp
// Umbrella header that re-exports the main library building blocks.
//
// Include this file when you want a single import for tensors, modules,
// activations, losses, sequential models, and optimizers.

// Neural Network Library - Main Header
// Include this header to access all library components

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

// Convenience namespace alias
namespace nn {
    // All components are in the nn namespace
}
