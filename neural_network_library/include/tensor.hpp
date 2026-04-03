#pragma once

/**
 * @file tensor.hpp
 * @brief Core tensor type, math APIs, and lightweight autograd hooks.
 */

#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <functional>
#include <random>

namespace nn {

/// Dense n-dimensional tensor with optional gradient storage.
///
/// Tensor owns a flat contiguous buffer and a shape vector describing how that
/// buffer should be interpreted. Most math operations return a new Tensor and,
/// when gradients are enabled, attach lightweight autograd metadata so backward()
/// can propagate gradients to upstream tensors.
class Tensor {
public:
    /// Element type stored in the tensor buffer.
    using value_type = double;

    /// Shape representation where each entry is the size of one dimension.
    using shape_type = std::vector<size_t>;
    
private:
    // Flat storage in row-major order.
    std::vector<value_type> data_;

    // Logical tensor dimensions.
    shape_type shape_;

    // Gradient buffer allocated lazily when needed.
    std::shared_ptr<Tensor> grad_;

    // Whether operations involving this tensor should record autograd state.
    bool requires_grad_;

    // Backward callback that applies this tensor's local gradient rule.
    std::function<void()> backward_fn_;

    // Upstream tensors that produced this tensor during the forward pass.
    std::vector<std::shared_ptr<Tensor>> parents_;

    /// Recursively executes the stored backward function and visits parents.
    void backward_impl();
    
public:
    /// Creates a tensor with the given shape and initializes all values to zero.
    explicit Tensor(const shape_type& shape, bool requires_grad = false);

    /// Creates a tensor from explicit data and validates that shape matches size.
    Tensor(const std::vector<value_type>& data, const shape_type& shape, 
           bool requires_grad = false);

    /// Copies tensor values and autograd metadata.
    Tensor(const Tensor& other);

    /// Moves tensor storage and autograd metadata.
    Tensor(Tensor&& other) noexcept;

    /// Copy assignment preserving value semantics.
    Tensor& operator=(const Tensor& other);

    /// Move assignment for efficient transfers.
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor() = default;
    
    /// Returns the tensor shape.
    const shape_type& shape() const { return shape_; }

    /// Returns the number of dimensions.
    size_t ndim() const { return shape_.size(); }

    /// Returns the total number of stored elements.
    size_t size() const;

    /// Returns mutable access to the raw contiguous storage.
    value_type* data() { return data_.data(); }

    /// Returns read-only access to the raw contiguous storage.
    const value_type* data() const { return data_.data(); }

    /// Reports whether this tensor participates in gradient tracking.
    bool requires_grad() const { return requires_grad_; }

    /// Enables or disables gradient tracking for future operations.
    void set_requires_grad(bool value) { requires_grad_ = value; }
    
    /// Returns mutable access by flat index.
    value_type& operator[](size_t idx) { return data_[idx]; }

    /// Returns read-only access by flat index.
    const value_type& operator[](size_t idx) const { return data_[idx]; }

    /// Returns mutable access using multi-dimensional coordinates.
    value_type& at(const std::vector<size_t>& indices);

    /// Returns read-only access using multi-dimensional coordinates.
    const value_type& at(const std::vector<size_t>& indices) const;
    
    /// Performs element-wise addition with another tensor of the same shape.
    Tensor operator+(const Tensor& other) const;

    /// Performs element-wise subtraction with another tensor of the same shape.
    Tensor operator-(const Tensor& other) const;

    /// Performs element-wise multiplication with another tensor of the same shape.
    Tensor operator*(const Tensor& other) const;

    /// Performs element-wise division with another tensor of the same shape.
    Tensor operator/(const Tensor& other) const;

    /// Adds a scalar to every tensor element.
    Tensor operator+(value_type scalar) const;

    /// Subtracts a scalar from every tensor element.
    Tensor operator-(value_type scalar) const;

    /// Multiplies every tensor element by a scalar.
    Tensor operator*(value_type scalar) const;

    /// Divides every tensor element by a scalar.
    Tensor operator/(value_type scalar) const;

    /// In-place element-wise addition.
    Tensor& operator+=(const Tensor& other);

    /// In-place element-wise subtraction.
    Tensor& operator-=(const Tensor& other);

    /// In-place element-wise multiplication.
    Tensor& operator*=(const Tensor& other);

    /// In-place scalar multiplication.
    Tensor& operator*=(value_type scalar);
    
    /// Multiplies two 2D tensors using matrix multiplication rules.
    Tensor matmul(const Tensor& other) const;

    /// Transposes a 2D tensor.
    Tensor transpose() const;
    
    /// Sums all tensor elements into a rank-1 tensor of shape {1}.
    Tensor sum() const;

    /// Sums tensor elements along a specific axis.
    Tensor sum(int axis) const;

    /// Computes the mean of all tensor elements.
    Tensor mean() const;

    /// Computes the mean of tensor elements along a specific axis.
    Tensor mean(int axis) const;

    /// Returns the maximum stored value.
    value_type max() const;

    /// Returns the minimum stored value.
    value_type min() const;
    
    /// Returns a new tensor view-like copy with the requested shape.
    Tensor reshape(const shape_type& new_shape) const;

    /// Flattens the tensor into a single dimension.
    Tensor flatten() const;

    /// Returns a slice along the given axis.
    Tensor slice(size_t start, size_t end, size_t axis = 0) const;
    
    /// Creates a zero-filled tensor.
    static Tensor zeros(const shape_type& shape, bool requires_grad = false);

    /// Creates a one-filled tensor.
    static Tensor ones(const shape_type& shape, bool requires_grad = false);

    /// Creates a tensor with samples from a standard normal distribution.
    static Tensor randn(const shape_type& shape, bool requires_grad = false);

    /// Creates a tensor with samples from a uniform distribution.
    static Tensor uniform(const shape_type& shape, value_type low = 0.0, 
                         value_type high = 1.0, bool requires_grad = false);
    
    /// Returns mutable access to the gradient tensor.
    Tensor& grad() { return *grad_; }

    /// Returns read-only access to the gradient tensor.
    const Tensor& grad() const { return *grad_; }

    /// Reports whether a gradient tensor has been allocated.
    bool has_grad() const { return grad_ != nullptr; }

    /// Returns the underlying shared pointer used for gradient storage.
    std::shared_ptr<Tensor> grad_ptr() const { return grad_; }

    /// Ensures that the gradient tensor exists.
    void ensure_grad();

    /// Stores the local backward rule and references to parent tensors.
    void set_autograd(std::function<void()> backward_fn,
                      std::vector<std::shared_ptr<Tensor>> parents);

    /// Creates a shared copy used when wiring simple autograd graphs.
    static std::shared_ptr<Tensor> alias(const Tensor& tensor);

    /// Starts backpropagation using a gradient of ones at this tensor.
    void backward();

    /// Resets the gradient buffer to zeros.
    void zero_grad();
    
    /// Prints a compact human-readable tensor summary.
    void print() const;

    /// Streams a compact tensor representation to an output stream.
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t);
    
private:
    /// Converts multi-dimensional coordinates into the flat buffer index.
    size_t compute_index(const std::vector<size_t>& indices) const;

    /// Throws when shapes are incompatible for element-wise operations.
    void check_shape_compatible(const Tensor& other) const;

    /// Allocates a zero-initialized gradient tensor matching this tensor's shape.
    void allocate_grad();
};

} // namespace nn
