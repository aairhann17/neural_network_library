#include "tensor.hpp"

// tensor.cpp
// Implementation of Tensor math, shape utilities, and lightweight autograd.
//
// High-level flow:
// - Each forward op computes output values.
// - If gradients are required, the op stores parent references and a
//   backward callback describing local gradient propagation.
// - backward() executes local propagation, then recursively traverses parents.

#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <memory>
#include <string>
#include <unordered_set>

namespace nn {

namespace {

std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

std::vector<size_t> remove_axis_shape(const std::vector<size_t>& shape, size_t axis) {
    std::vector<size_t> out_shape;
    out_shape.reserve(shape.size() > 0 ? shape.size() - 1 : 0);
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != axis) {
            out_shape.push_back(shape[i]);
        }
    }
    if (out_shape.empty()) {
        out_shape.push_back(1);
    }
    return out_shape;
}

size_t out_index_without_axis(size_t flat_idx,
                              const std::vector<size_t>& shape,
                              const std::vector<size_t>& in_strides,
                              size_t axis,
                              const std::vector<size_t>& out_strides,
                              size_t out_ndim_effective) {
    size_t out_idx = 0;
    size_t out_dim = 0;

    for (size_t dim = 0; dim < shape.size(); ++dim) {
        const size_t coord = (flat_idx / in_strides[dim]) % shape[dim];
        if (dim == axis) {
            continue;
        }
        if (out_ndim_effective > 0) {
            out_idx += coord * out_strides[out_dim];
        }
        ++out_dim;
    }

    return out_idx;
}

} // namespace

// ============ Constructors ============

Tensor::Tensor(const shape_type& shape, bool requires_grad)
    : shape_(shape), grad_(nullptr), requires_grad_(requires_grad) {
    
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
    : shape_(shape), grad_(nullptr), requires_grad_(requires_grad) {
    
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
            grad_(other.grad_),
            requires_grad_(other.requires_grad_),
            backward_fn_(other.backward_fn_),
            parents_(other.parents_) {

        if (requires_grad_ && !grad_) {
                allocate_grad();
        }
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(std::move(other.data_)),
      shape_(std::move(other.shape_)),
            grad_(std::move(other.grad_)),
            requires_grad_(other.requires_grad_),
            backward_fn_(std::move(other.backward_fn_)),
            parents_(std::move(other.parents_)) {}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        data_ = other.data_;
        shape_ = other.shape_;
        grad_ = other.grad_;
        requires_grad_ = other.requires_grad_;
        backward_fn_ = other.backward_fn_;
        parents_ = other.parents_;
        if (requires_grad_ && !grad_) {
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
        backward_fn_ = std::move(other.backward_fn_);
        parents_ = std::move(other.parents_);
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

    if (result.requires_grad_) {
        auto left = Tensor::alias(*this);
        auto right = Tensor::alias(other);
        auto out_grad = result.grad_;
        result.parents_ = {left, right};
        result.backward_fn_ = [out_grad, left, right]() {
            if (!out_grad) {
                return;
            }
            if (left->requires_grad_) {
                if (!left->grad_) {
                    left->allocate_grad();
                }
                for (size_t i = 0; i < left->data_.size(); ++i) {
                    left->grad_->data_[i] += out_grad->data_[i];
                }
            }
            if (right->requires_grad_) {
                if (!right->grad_) {
                    right->allocate_grad();
                }
                for (size_t i = 0; i < right->data_.size(); ++i) {
                    right->grad_->data_[i] += out_grad->data_[i];
                }
            }
        };
    }
    
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    check_shape_compatible(other);
    
    Tensor result(shape_, requires_grad_ || other.requires_grad_);
    
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }

    if (result.requires_grad_) {
        auto left = Tensor::alias(*this);
        auto right = Tensor::alias(other);
        auto out_grad = result.grad_;
        result.parents_ = {left, right};
        result.backward_fn_ = [out_grad, left, right]() {
            if (!out_grad) {
                return;
            }
            if (left->requires_grad_) {
                if (!left->grad_) {
                    left->allocate_grad();
                }
                for (size_t i = 0; i < left->data_.size(); ++i) {
                    left->grad_->data_[i] += out_grad->data_[i];
                }
            }
            if (right->requires_grad_) {
                if (!right->grad_) {
                    right->allocate_grad();
                }
                for (size_t i = 0; i < right->data_.size(); ++i) {
                    right->grad_->data_[i] -= out_grad->data_[i];
                }
            }
        };
    }
    
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    check_shape_compatible(other);
    
    Tensor result(shape_, requires_grad_ || other.requires_grad_);
    
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }

    if (result.requires_grad_) {
        auto left = Tensor::alias(*this);
        auto right = Tensor::alias(other);
        auto out_grad = result.grad_;
        result.parents_ = {left, right};
        result.backward_fn_ = [out_grad, left, right]() {
            if (!out_grad) {
                return;
            }
            if (left->requires_grad_) {
                if (!left->grad_) {
                    left->allocate_grad();
                }
                for (size_t i = 0; i < left->data_.size(); ++i) {
                    left->grad_->data_[i] += out_grad->data_[i] * right->data_[i];
                }
            }
            if (right->requires_grad_) {
                if (!right->grad_) {
                    right->allocate_grad();
                }
                for (size_t i = 0; i < right->data_.size(); ++i) {
                    right->grad_->data_[i] += out_grad->data_[i] * left->data_[i];
                }
            }
        };
    }
    
    return result;
}

Tensor Tensor::operator/(const Tensor& other) const {
    check_shape_compatible(other);

    Tensor result(shape_, requires_grad_ || other.requires_grad_);

    for (size_t i = 0; i < data_.size(); ++i) {
        if (other.data_[i] == 0.0) {
            throw std::runtime_error("Division by zero in tensor division");
        }
        result.data_[i] = data_[i] / other.data_[i];
    }

    if (result.requires_grad_) {
        auto left = Tensor::alias(*this);
        auto right = Tensor::alias(other);
        auto out_grad = result.grad_;
        result.parents_ = {left, right};
        result.backward_fn_ = [out_grad, left, right]() {
            if (!out_grad) {
                return;
            }
            if (left->requires_grad_) {
                if (!left->grad_) {
                    left->allocate_grad();
                }
                for (size_t i = 0; i < left->data_.size(); ++i) {
                    left->grad_->data_[i] += out_grad->data_[i] / right->data_[i];
                }
            }
            if (right->requires_grad_) {
                if (!right->grad_) {
                    right->allocate_grad();
                }
                for (size_t i = 0; i < right->data_.size(); ++i) {
                    const value_type denom = right->data_[i] * right->data_[i];
                    right->grad_->data_[i] -= out_grad->data_[i] * left->data_[i] / denom;
                }
            }
        };
    }

    return result;
}

Tensor Tensor::operator+(value_type scalar) const {
    Tensor result(shape_, requires_grad_);

    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + scalar;
    }

    if (result.requires_grad_) {
        auto left = Tensor::alias(*this);
        auto out_grad = result.grad_;
        result.parents_ = {left};
        result.backward_fn_ = [out_grad, left]() {
            if (!out_grad || !left->requires_grad_) {
                return;
            }
            if (!left->grad_) {
                left->allocate_grad();
            }
            for (size_t i = 0; i < left->data_.size(); ++i) {
                left->grad_->data_[i] += out_grad->data_[i];
            }
        };
    }

    return result;
}

Tensor Tensor::operator-(value_type scalar) const {
    Tensor result(shape_, requires_grad_);

    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - scalar;
    }

    if (result.requires_grad_) {
        auto left = Tensor::alias(*this);
        auto out_grad = result.grad_;
        result.parents_ = {left};
        result.backward_fn_ = [out_grad, left]() {
            if (!out_grad || !left->requires_grad_) {
                return;
            }
            if (!left->grad_) {
                left->allocate_grad();
            }
            for (size_t i = 0; i < left->data_.size(); ++i) {
                left->grad_->data_[i] += out_grad->data_[i];
            }
        };
    }

    return result;
}

Tensor Tensor::operator*(value_type scalar) const {
    Tensor result(shape_, requires_grad_);
    
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }

    if (result.requires_grad_) {
        auto left = Tensor::alias(*this);
        auto out_grad = result.grad_;
        result.parents_ = {left};
        result.backward_fn_ = [out_grad, left, scalar]() {
            if (!out_grad || !left->requires_grad_) {
                return;
            }
            if (!left->grad_) {
                left->allocate_grad();
            }
            for (size_t i = 0; i < left->data_.size(); ++i) {
                left->grad_->data_[i] += out_grad->data_[i] * scalar;
            }
        };
    }

    return result;
}

Tensor Tensor::operator/(value_type scalar) const {
    if (scalar == 0.0) {
        throw std::runtime_error("Division by zero scalar");
    }

    Tensor result(shape_, requires_grad_);

    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] / scalar;
    }

    if (result.requires_grad_) {
        auto left = Tensor::alias(*this);
        auto out_grad = result.grad_;
        result.parents_ = {left};
        result.backward_fn_ = [out_grad, left, scalar]() {
            if (!out_grad || !left->requires_grad_) {
                return;
            }
            if (!left->grad_) {
                left->allocate_grad();
            }
            for (size_t i = 0; i < left->data_.size(); ++i) {
                left->grad_->data_[i] += out_grad->data_[i] / scalar;
            }
        };
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

Tensor& Tensor::operator-=(const Tensor& other) {
    check_shape_compatible(other);

    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] -= other.data_[i];
    }

    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
    check_shape_compatible(other);

    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] *= other.data_[i];
    }

    return *this;
}

Tensor& Tensor::operator*=(value_type scalar) {
    for (auto& value : data_) {
        value *= scalar;
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

    if (result.requires_grad_) {
        auto left = Tensor::alias(*this);
        auto right = Tensor::alias(other);
        auto out_grad = result.grad_;
        result.parents_ = {left, right};
        result.backward_fn_ = [out_grad, left, right]() {
            if (!out_grad) {
                return;
            }

            const size_t m_local = left->shape_[0];
            const size_t k_local = left->shape_[1];
            const size_t n_local = right->shape_[1];

            if (left->requires_grad_) {
                if (!left->grad_) {
                    left->allocate_grad();
                }

                for (size_t i = 0; i < m_local; ++i) {
                    for (size_t p = 0; p < k_local; ++p) {
                        value_type acc = 0.0;
                        for (size_t j = 0; j < n_local; ++j) {
                            acc += out_grad->data_[i * n_local + j] * right->data_[p * n_local + j];
                        }
                        left->grad_->data_[i * k_local + p] += acc;
                    }
                }
            }

            if (right->requires_grad_) {
                if (!right->grad_) {
                    right->allocate_grad();
                }

                for (size_t p = 0; p < k_local; ++p) {
                    for (size_t j = 0; j < n_local; ++j) {
                        value_type acc = 0.0;
                        for (size_t i = 0; i < m_local; ++i) {
                            acc += left->data_[i * k_local + p] * out_grad->data_[i * n_local + j];
                        }
                        right->grad_->data_[p * n_local + j] += acc;
                    }
                }
            }
        };
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

    if (result.requires_grad_) {
        auto input = Tensor::alias(*this);
        auto out_grad = result.grad_;
        result.parents_ = {input};
        result.backward_fn_ = [input, out_grad, rows, cols]() {
            if (!out_grad || !input->requires_grad_) {
                return;
            }
            if (!input->grad_) {
                input->allocate_grad();
            }
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    input->grad_->data_[i * cols + j] += out_grad->data_[j * rows + i];
                }
            }
        };
    }
    
    return result;
}

// ============ Reduction Operations ============

Tensor Tensor::sum() const {
    value_type total = std::accumulate(data_.begin(), data_.end(), 0.0);
    Tensor result({total}, {1}, requires_grad_);

    if (result.requires_grad_) {
        auto input = Tensor::alias(*this);
        auto out_grad = result.grad_;
        result.parents_ = {input};
        result.backward_fn_ = [out_grad, input]() {
            if (!out_grad || !input->requires_grad_) {
                return;
            }
            if (!input->grad_) {
                input->allocate_grad();
            }

            const value_type upstream = out_grad->data_[0];
            for (size_t i = 0; i < input->data_.size(); ++i) {
                input->grad_->data_[i] += upstream;
            }
        };
    }

    return result;
}

Tensor Tensor::sum(int axis) const {
    if (shape_.empty()) {
        throw std::invalid_argument("Cannot reduce axis on scalar-like empty shape");
    }

    int normalized_axis = axis;
    if (normalized_axis < 0) {
        normalized_axis += static_cast<int>(shape_.size());
    }
    if (normalized_axis < 0 || normalized_axis >= static_cast<int>(shape_.size())) {
        throw std::out_of_range("Axis out of range in sum(axis)");
    }

    const size_t axis_u = static_cast<size_t>(normalized_axis);
    const auto out_shape = remove_axis_shape(shape_, axis_u);
    Tensor result(out_shape, requires_grad_);

    const auto in_strides = compute_strides(shape_);
    const size_t out_ndim_effective = shape_.size() > 1 ? shape_.size() - 1 : 0;
    const auto out_strides = out_ndim_effective > 0 ? compute_strides(out_shape) : std::vector<size_t>{};

    for (size_t in_idx = 0; in_idx < data_.size(); ++in_idx) {
        const size_t out_idx = out_index_without_axis(
            in_idx,
            shape_,
            in_strides,
            axis_u,
            out_strides,
            out_ndim_effective
        );
        result.data_[out_idx] += data_[in_idx];
    }

    if (result.requires_grad_) {
        auto input = Tensor::alias(*this);
        auto out_grad = result.grad_;
        result.parents_ = {input};
        result.backward_fn_ = [out_grad, input, axis_u]() {
            if (!out_grad || !input->requires_grad_) {
                return;
            }
            if (!input->grad_) {
                input->allocate_grad();
            }

            const auto in_strides_local = compute_strides(input->shape_);
            const size_t out_ndim_effective_local = input->shape_.size() > 1 ? input->shape_.size() - 1 : 0;
            const auto out_shape_local = remove_axis_shape(input->shape_, axis_u);
            const auto out_strides_local = out_ndim_effective_local > 0 ? compute_strides(out_shape_local) : std::vector<size_t>{};

            for (size_t in_idx = 0; in_idx < input->data_.size(); ++in_idx) {
                const size_t out_idx = out_index_without_axis(
                    in_idx,
                    input->shape_,
                    in_strides_local,
                    axis_u,
                    out_strides_local,
                    out_ndim_effective_local
                );
                input->grad_->data_[in_idx] += out_grad->data_[out_idx];
            }
        };
    }

    return result;
}

Tensor Tensor::mean() const {
    value_type avg = std::accumulate(data_.begin(), data_.end(), 0.0) /
                     static_cast<value_type>(data_.size());
    Tensor result({avg}, {1}, requires_grad_);

    if (result.requires_grad_) {
        auto input = Tensor::alias(*this);
        auto out_grad = result.grad_;
        result.parents_ = {input};
        result.backward_fn_ = [out_grad, input]() {
            if (!out_grad || !input->requires_grad_) {
                return;
            }
            if (!input->grad_) {
                input->allocate_grad();
            }

            const value_type scale = out_grad->data_[0] /
                                     static_cast<value_type>(input->data_.size());
            for (size_t i = 0; i < input->data_.size(); ++i) {
                input->grad_->data_[i] += scale;
            }
        };
    }

    return result;
}

Tensor Tensor::mean(int axis) const {
    if (shape_.empty()) {
        throw std::invalid_argument("Cannot reduce axis on scalar-like empty shape");
    }

    int normalized_axis = axis;
    if (normalized_axis < 0) {
        normalized_axis += static_cast<int>(shape_.size());
    }
    if (normalized_axis < 0 || normalized_axis >= static_cast<int>(shape_.size())) {
        throw std::out_of_range("Axis out of range in mean(axis)");
    }

    const size_t axis_u = static_cast<size_t>(normalized_axis);
    Tensor summed = sum(axis_u);
    const value_type denom = static_cast<value_type>(shape_[axis_u]);
    return summed / denom;
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

void Tensor::ensure_grad() {
    if (!grad_) {
        allocate_grad();
    }
}

void Tensor::set_autograd(std::function<void()> backward_fn,
                          std::vector<std::shared_ptr<Tensor>> parents) {
    backward_fn_ = std::move(backward_fn);
    parents_ = std::move(parents);
}

std::shared_ptr<Tensor> Tensor::alias(const Tensor& tensor) {
    return std::make_shared<Tensor>(tensor);
}

void Tensor::backward() {
    if (!requires_grad_) {
        throw std::runtime_error("Cannot backward on tensor with requires_grad=false");
    }
    if (!grad_) allocate_grad();
    std::fill(grad_->data_.begin(), grad_->data_.end(), 1.0);
    backward_impl();
}

void Tensor::backward_impl() {
    static thread_local std::unordered_set<const Tensor*> active;
    if (active.find(this) != active.end()) {
        return;
    }

    active.insert(this);
    if (backward_fn_) backward_fn_();
    for (auto& parent : parents_) {
        if (parent && parent->requires_grad_) {
            parent->backward_impl();
        }
    }
    active.erase(this);
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
