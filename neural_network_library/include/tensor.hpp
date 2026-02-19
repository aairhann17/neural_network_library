#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <functional>
#include <random>

namespace nn {

class Tensor {
public:
    using value_type = double;
    using shape_type = std::vector<size_t>;
    
private:
    std::vector<value_type> data_;
    shape_type shape_;
    std::shared_ptr<Tensor> grad_;
    bool requires_grad_;
    std::function<void()> backward_fn_;
    std::vector<std::shared_ptr<Tensor>> parents_;
    
public:
    // Constructors
    explicit Tensor(const shape_type& shape, bool requires_grad = false);
    Tensor(const std::vector<value_type>& data, const shape_type& shape, 
           bool requires_grad = false);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor() = default;
    
    // Basic Properties
    const shape_type& shape() const { return shape_; }
    size_t ndim() const { return shape_.size(); }
    size_t size() const;
    value_type* data() { return data_.data(); }
    const value_type* data() const { return data_.data(); }
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool value) { requires_grad_ = value; }
    
    // Element Access
    value_type& operator[](size_t idx) { return data_[idx]; }
    const value_type& operator[](size_t idx) const { return data_[idx]; }
    value_type& at(const std::vector<size_t>& indices);
    const value_type& at(const std::vector<size_t>& indices) const;
    
    // Arithmetic Operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator+(value_type scalar) const;
    Tensor operator-(value_type scalar) const;
    Tensor operator*(value_type scalar) const;
    Tensor operator/(value_type scalar) const;
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator*=(value_type scalar);
    
    // Matrix Operations
    Tensor matmul(const Tensor& other) const;
    Tensor transpose() const;
    
    // Reduction Operations
    Tensor sum() const;
    Tensor sum(int axis) const;
    Tensor mean() const;
    Tensor mean(int axis) const;
    value_type max() const;
    value_type min() const;
    
    // Shape Manipulation
    Tensor reshape(const shape_type& new_shape) const;
    Tensor flatten() const;
    Tensor slice(size_t start, size_t end, size_t axis = 0) const;
    
    // Initialization
    static Tensor zeros(const shape_type& shape, bool requires_grad = false);
    static Tensor ones(const shape_type& shape, bool requires_grad = false);
    static Tensor randn(const shape_type& shape, bool requires_grad = false);
    static Tensor uniform(const shape_type& shape, value_type low = 0.0, 
                         value_type high = 1.0, bool requires_grad = false);
    
    // Autograd
    Tensor& grad() { return *grad_; }
    const Tensor& grad() const { return *grad_; }
    bool has_grad() const { return grad_ != nullptr; }
    void backward();
    void zero_grad();
    
    // Utilities
    void print() const;
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t);
    
private:
    size_t compute_index(const std::vector<size_t>& indices) const;
    void check_shape_compatible(const Tensor& other) const;
    void allocate_grad();
};

} // namespace nn