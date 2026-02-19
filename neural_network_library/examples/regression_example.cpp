#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include "tensor.hpp"
#include "module.hpp"
#include "sequential.hpp"
#include "optimizer.hpp"
#include "losses.hpp"

using namespace nn;

// Generate synthetic regression data: y = 3x + 2 + noise
void generate_regression_data(std::vector<Tensor>& inputs, std::vector<Tensor>& targets, 
                               int num_samples = 100) {
    
    for (int i = 0; i < num_samples; ++i) {
        double x = static_cast<double>(i) / num_samples * 10.0 - 5.0; // [-5, 5]
        double y = 3.0 * x + 2.0 + (static_cast<double>(rand()) / RAND_MAX - 0.5) * 2.0;
        
        inputs.push_back(Tensor({x}, {1, 1}, false));
        targets.push_back(Tensor({y}, {1, 1}, false));
    }
}

int main() {
    std::cout << "=== Linear Regression with Neural Network ===" << std::endl;
    std::cout << "Learning the function: y = 3x + 2\n" << std::endl;
    
    // Create a simple neural network for regression
    // Architecture: 1 -> 10 -> 1
    auto model = std::make_shared<Sequential>();
    model->add(std::make_shared<Linear>(1, 10));
    model->add(std::make_shared<ReLU>());
    model->add(std::make_shared<Linear>(10, 1));
    
    // Create optimizer
    auto params = model->parameters();
    SGD optimizer(params, 0.001, 0.9); // learning rate = 0.001, momentum = 0.9
    
    // Generate dataset
    std::vector<Tensor> inputs, targets;
    generate_regression_data(inputs, targets, 100);
    
    std::cout << "Model architecture:" << std::endl;
    std::cout << "  Layer 1: Linear(1 -> 10) + ReLU" << std::endl;
    std::cout << "  Layer 2: Linear(10 -> 1)" << std::endl;
    std::cout << "Dataset size: " << inputs.size() << " samples" << std::endl;
    std::cout << "\nTraining..." << std::endl;
    
    // Training loop
    int num_epochs = 1000;
    int print_every = 100;
    
    model->train();
    
    for (int epoch = 1; epoch <= num_epochs; ++epoch) {
        double total_loss = 0.0;
        
        for (size_t i = 0; i < inputs.size(); ++i) {
            optimizer.zero_grad();
            
            Tensor prediction = model->forward(inputs[i]);
            Tensor loss = losses::mse_loss(prediction, targets[i]);
            total_loss += loss[0];
            
            // In complete implementation: loss.backward()
            optimizer.step();
        }
        
        double avg_loss = total_loss / inputs.size();
        
        if (epoch % print_every == 0 || epoch == 1) {
            std::cout << "Epoch " << epoch << "/" << num_epochs 
                      << " - MSE Loss: " << avg_loss << std::endl;
        }
    }
    
    std::cout << "\n=== Testing predictions ===" << std::endl;
    model->eval();
    
    std::vector<double> test_inputs = {-5.0, -2.5, 0.0, 2.5, 5.0};
    for (double x : test_inputs) {
        Tensor input({x}, {1, 1}, false);
        Tensor prediction = model->forward(input);
        double expected = 3.0 * x + 2.0;
        
        std::cout << "Input: " << x 
                  << " -> Prediction: " << prediction[0]
                  << " (Expected: " << expected << ")" << std::endl;
    }
    
    std::cout << "\n=== Regression demo completed! ===" << std::endl;
    
    return 0;
}
