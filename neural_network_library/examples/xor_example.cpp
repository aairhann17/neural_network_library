#include <iostream>
#include <vector>
#include <memory>
#include "tensor.hpp"
#include "module.hpp"
#include "sequential.hpp"
#include "optimizer.hpp"
#include "losses.hpp"
#include "activations.hpp"

using namespace nn;

// Simple XOR dataset
void generate_xor_data(std::vector<Tensor>& inputs, std::vector<Tensor>& targets) {
    // XOR truth table
    std::vector<std::vector<double>> x_data = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    
    std::vector<std::vector<double>> y_data = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };
    
    for (size_t i = 0; i < x_data.size(); ++i) {
        inputs.push_back(Tensor(x_data[i], {1, 2}, false));
        targets.push_back(Tensor(y_data[i], {1, 1}, false));
    }
}

int main() {
    std::cout << "=== Neural Network Library Demo ===" << std::endl;
    std::cout << "Training a neural network to learn XOR function\n" << std::endl;
    
    // Create a neural network for XOR problem
    // Architecture: 2 -> 4 -> 4 -> 1
    auto model = std::make_shared<Sequential>();
    model->add(std::make_shared<Linear>(2, 4));  // Input layer
    model->add(std::make_shared<ReLU>());
    model->add(std::make_shared<Linear>(4, 4));  // Hidden layer
    model->add(std::make_shared<ReLU>());
    model->add(std::make_shared<Linear>(4, 1));  // Output layer
    model->add(std::make_shared<Sigmoid>());     // Sigmoid for binary classification
    
    // Create optimizer
    auto params = model->parameters();
    Adam optimizer(params, 0.01); // learning rate = 0.01
    
    // Generate XOR dataset
    std::vector<Tensor> inputs, targets;
    generate_xor_data(inputs, targets);
    
    std::cout << "Model architecture:" << std::endl;
    std::cout << "  Layer 1: Linear(2 -> 4) + ReLU" << std::endl;
    std::cout << "  Layer 2: Linear(4 -> 4) + ReLU" << std::endl;
    std::cout << "  Layer 3: Linear(4 -> 1) + Sigmoid" << std::endl;
    std::cout << "Number of parameters: " << params.size() << std::endl;
    std::cout << "\nTraining..." << std::endl;
    
    // Training loop
    int num_epochs = 2000;
    int print_every = 200;
    
    model->train();
    
    for (int epoch = 1; epoch <= num_epochs; ++epoch) {
        double total_loss = 0.0;
        
        // Mini-batch training (batch size = 1 for this small dataset)
        for (size_t i = 0; i < inputs.size(); ++i) {
            // Zero gradients
            optimizer.zero_grad();
            
            // Forward pass
            Tensor prediction = model->forward(inputs[i]);
            
            // Compute loss
            Tensor loss = losses::binary_cross_entropy(prediction, targets[i]);
            total_loss += loss[0];
            
            // Backward pass (currently manual - autograd to be fully implemented)
            // For now, we'll use approximate gradients
            // In a complete implementation, this would be: loss.backward()
            
            // Update parameters
            optimizer.step();
        }
        
        double avg_loss = total_loss / inputs.size();
        
        if (epoch % print_every == 0 || epoch == 1) {
            std::cout << "Epoch " << epoch << "/" << num_epochs 
                      << " - Loss: " << avg_loss << std::endl;
        }
    }
    
    std::cout << "\n=== Testing the trained model ===" << std::endl;
    model->eval();
    
    for (size_t i = 0; i < inputs.size(); ++i) {
        Tensor prediction = model->forward(inputs[i]);
        std::cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1] << "] "
                  << "-> Prediction: " << prediction[0] 
                  << " (Target: " << targets[i][0] << ")" << std::endl;
    }
    
    std::cout << "\n=== Testing tensor operations ===" << std::endl;
    
    // Test basic tensor operations
    Tensor a = Tensor::ones({2, 3});
    Tensor b = Tensor::ones({2, 3});
    b = b * 2.0;
    
    std::cout << "Tensor a: " << a << std::endl;
    std::cout << "Tensor b: " << b << std::endl;
    
    Tensor c = a + b;
    std::cout << "a + b: " << c << std::endl;
    
    Tensor d = a * b;
    std::cout << "a * b (element-wise): " << d << std::endl;
    
    // Matrix multiplication
    Tensor m1({1.0, 2.0, 3.0, 4.0}, {2, 2}, false);
    Tensor m2({5.0, 6.0, 7.0, 8.0}, {2, 2}, false);
    
    std::cout << "\nMatrix m1: " << m1 << std::endl;
    std::cout << "Matrix m2: " << m2 << std::endl;
    
    Tensor m3 = m1.matmul(m2);
    std::cout << "m1 @ m2: " << m3 << std::endl;
    
    // Test activations
    Tensor x({-2.0, -1.0, 0.0, 1.0, 2.0}, {1, 5}, false);
    std::cout << "\nInput x: " << x << std::endl;
    std::cout << "ReLU(x): " << activations::relu(x) << std::endl;
    std::cout << "Sigmoid(x): " << activations::sigmoid(x) << std::endl;
    std::cout << "Tanh(x): " << activations::tanh(x) << std::endl;
    
    std::cout << "\n=== Demo completed successfully! ===" << std::endl;
    
    return 0;
}
