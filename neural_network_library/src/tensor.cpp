#include "tensor.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <memory>
#include <string>

namespace nn {

// ============ Constructors ============

Tensor::Tensor(const shape_type& shape, bool requires_grad)
    : shape_(shape), requires_grad_(requires_grad), grad_(nullptr) {
    
    size_t total_size = 1;
    for (auto dim : shape) {
        if (dim == 0) {
            throw std::invalid_argument("Shape dimensions must be > 0");
        }
        total_size *= dim;
    }
    
    data_.resize(total_size, 0.0);
    
    if (requires_grad_) {
        allocate_grad();
    }
}

Tensor::Tensor(const std::vector<value_type>& data, const shape_type& shape,
               bool requires_grad)
    : shape_(shape), requires_grad_(requires_grad), grad_(nullptr) {
    
    size_t expected_size = 1;
    for (auto dim : shape) expected_size *= dim;
    
    if (data.size() != expected_size) {
        throw std::invalid_argument(
            "Data size doesn't match shape. Expected: " + 
            std::to_string(expected_size) + ", got: " + 
            std::to_string(data.size())
        );
    }
    
    data_ = data;
    
    if (requires_grad_) {
        allocate_grad();
    }
}

Tensor::Tensor(const Tensor& other)
    : data_(other.data_),
      shape_(other.shape_),
      requires_grad_(other.requires_grad_),
      grad_(nullptr) {
    
    if (requires_grad_) {
        allocate_grad();
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(std::move(other.data_)),
      shape_(std::move(other.shape_)),
      requires_grad_(other.requires_grad_),
      grad_(std::move(other.grad_)) {}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        data_ = other.data_;
        shape_ = other.shape_;
        requires_grad_ = other.requires_grad_;
        grad_ = nullptr;
        if (requires_grad_) {
            allocate_grad();
        }
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        data_ = std::move(other.data_);
        shape_ = std::move(other.shape_);
        requires_grad_ = other.requires_grad_;
        grad_ = std::move(other.grad_);
    }
    return *this;
}

// ============ Basic Properties ============

size_t Tensor::size() const {
    return data_.size();
}

// ============ Element Access ============

Tensor::value_type& Tensor::at(const std::vector<size_t>& indices) {
    return data_[compute_index(indices)];
}

const Tensor::value_type& Tensor::at(const std::vector<size_t>& indices) const {
    return data_[compute_index(indices)];
}

size_t Tensor::compute_index(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Number of indices doesn't match dimensions");
    }
    
    size_t index = 0;
    size_t stride = 1;
    
    for (int i = shape_.size() - 1; i >= 0; --i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of bounds");
        }
        index += indices[i] * stride;
        stride *= shape_[i];
    }
    
    return index;
}

// ============ Arithmetic Operations ============

Tensor Tensor::operator+(const Tensor& other) const {
    check_shape_compatible(other);
    
    Tensor result(shape_, requires_grad_ || other.requires_grad_);
    
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    check_shape_compatible(other);
    
    Tensor result(shape_, requires_grad_ || other.requires_grad_);
    
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    check_shape_compatible(other);
    
    Tensor result(shape_, requires_grad_ || other.requires_grad_);
    
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    
    return result;
}

Tensor Tensor::operator*(value_type scalar) const {
    Tensor result(shape_, requires_grad_);
    
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    
    return result;
}

Tensor& Tensor::operator+=(const Tensor& other) {
    check_shape_compatible(other);
    
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] += other.data_[i];
    }
    
    return *this;
}

// ============ Matrix Operations ============

Tensor Tensor::matmul(const Tensor& other) const {
    if (shape_.size() != 2 || other.shape_.size() != 2) {
        throw std::invalid_argument("matmul currently only supports 2D tensors");
    }
    
    size_t m = shape_[0];
    size_t k = shape_[1];
    size_t n = other.shape_[1];
    
    if (k != other.shape_[0]) {
        throw std::invalid_argument(
            "Invalid dimensions for matmul: (" + 
            std::to_string(m) + "x" + std::to_string(k) + 
            ") @ (" + std::to_string(other.shape_[0]) + "x" + 
            std::to_string(n) + ")"
        );
    }
    
    Tensor result({m, n}, requires_grad_ || other.requires_grad_);
    
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            value_type sum = 0.0;
            for (size_t p = 0; p < k; ++p) {
                sum += data_[i * k + p] * other.data_[p * n + j];
            }
            result.data_[i * n + j] = sum;
        }
    }
    
    return result;
}

Tensor Tensor::transpose() const {
    if (shape_.size() != 2) {
        throw std::invalid_argument("transpose only supports 2D tensors");
    }
    
    size_t rows = shape_[0];
    size_t cols = shape_[1];
    
    Tensor result({cols, rows}, requires_grad_);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data_[j * rows + i] = data_[i * cols + j];
        }
    }
    
    return result;
}

// ============ Reduction Operations ============

Tensor Tensor::sum() const {
    value_type total = std::accumulate(data_.begin(), data_.end(), 0.0);
    return Tensor({total}, {1}, requires_grad_);
}

Tensor Tensor::mean() const {
    value_type avg = sum().data_[0] / static_cast<value_type>(data_.size());
    return Tensor({avg}, {1}, requires_grad_);
}

Tensor::value_type Tensor::max() const {
    return *std::max_element(data_.begin(), data_.end());
}

Tensor::value_type Tensor::min() const {
    return *std::min_element(data_.begin(), data_.end());
}

// ============ Shape Manipulation ============

Tensor Tensor::reshape(const shape_type& new_shape) const {
    size_t new_size = 1;
    for (auto dim : new_shape) new_size *= dim;
    
    if (new_size != data_.size()) {
        throw std::invalid_argument("New shape size doesn't match tensor size");
    }
    
    return Tensor(data_, new_shape, requires_grad_);
}

Tensor Tensor::flatten() const {
    return reshape({data_.size()});
}

Tensor Tensor::slice(size_t start, size_t end, size_t axis) const {
    if (axis >= shape_.size()) {
        throw std::out_of_range("Axis out of range");
    }
    
    if (start >= end || end > shape_[axis]) {
        throw std::invalid_argument("Invalid slice range");
    }
    
    if (axis != 0) {
        throw std::runtime_error("Slicing only implemented for axis=0");
    }
    
    size_t row_size = data_.size() / shape_[0];
    size_t new_rows = end - start;
    
    shape_type new_shape = shape_;
    new_shape[0] = new_rows;
    
    std::vector<value_type> new_data(
        data_.begin() + start * row_size,
        data_.begin() + end * row_size
    );
    
    return Tensor(new_data, new_shape, requires_grad_);
}

// ============ Initialization ============

Tensor Tensor::zeros(const shape_type& shape, bool requires_grad) {
    return Tensor(shape, requires_grad);
}

Tensor Tensor::ones(const shape_type& shape, bool requires_grad) {
    Tensor t(shape, requires_grad);
    std::fill(t.data_.begin(), t.data_.end(), 1.0);
    return t;
}

Tensor Tensor::randn(const shape_type& shape, bool requires_grad) {
    Tensor t(shape, requires_grad);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<value_type> dist(0.0, 1.0);
    
    for (auto& val : t.data_) {
        val = dist(gen);
    }
    
    return t;
}

Tensor Tensor::uniform(const shape_type& shape, value_type low, 
                       value_type high, bool requires_grad) {
    Tensor t(shape, requires_grad);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<value_type> dist(low, high);
    
    for (auto& val : t.data_) {
        val = dist(gen);
    }
    
    return t;
}

// ============ Autograd ============

void Tensor::allocate_grad() {
    if (!grad_) {
        grad_ = std::make_shared<Tensor>(shape_, false);
    }
}

void Tensor::zero_grad() {
    if (grad_) {
        std::fill(grad_->data_.begin(), grad_->data_.end(), 0.0);
    }
}

void Tensor::backward() {
    if (!requires_grad_) {
        throw std::runtime_error("Cannot backward on tensor with requires_grad=false");
    }
    
    if (!grad_) {
        allocate_grad();
        std::fill(grad_->data_.begin(), grad_->data_.end(), 1.0);
    }
    
    if (backward_fn_) {
        backward_fn_();
    }
}

// ============ Utilities ============

void Tensor::check_shape_compatible(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Incompatible tensor shapes");
    }
}

void Tensor::print() const {
    std::cout << *this << std::endl;
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << "Tensor(shape=[";
    for (size_t i = 0; i < t.shape_.size(); ++i) {
        os << t.shape_[i];
        if (i < t.shape_.size() - 1) os << ", ";
    }
    os << "], data=[";
    
    size_t print_limit = std::min(t.data_.size(), size_t(10));
    for (size_t i = 0; i < print_limit; ++i) {
        os << std::fixed << std::setprecision(4) << t.data_[i];
        if (i < print_limit - 1) os << ", ";
    }
    if (t.data_.size() > print_limit) {
        os << ", ...";
    }
    os << "])";
    
    return os;
}

} // namespace nn