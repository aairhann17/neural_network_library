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

/** @ingroup tensor_api */
/// Dense n-dimensional tensor with optional gradient storage.
///
/// Tensor owns a flat contiguous buffer and a shape vector describing how that
/// buffer should be interpreted. Most math operations return a new Tensor and,
/// when gradients are enabled, attach lightweight autograd metadata so backward()
/// can propagate gradients to upstream tensors.
class Tensor {
public:
    /** @brief Element type stored in the tensor buffer. */
    using value_type = double;

    /** @brief Shape representation where each entry is one dimension size. */
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

    /** @brief Executes backward callback and recursively traverses parents. */
    void backward_impl();
    
public:
        /**
         * @brief Creates a zero-initialized tensor with the provided shape.
         * @param shape Tensor dimensions in row-major order.
         * @param requires_grad Whether operations involving this tensor should track gradients.
         * @throws std::invalid_argument If any dimension in shape is zero.
         */
    explicit Tensor(const shape_type& shape, bool requires_grad = false);

        /**
         * @brief Creates a tensor from explicit data and shape.
         * @param data Flat row-major tensor values.
         * @param shape Tensor dimensions used to interpret data.
         * @param requires_grad Whether operations involving this tensor should track gradients.
         * @throws std::invalid_argument If data size does not match shape product.
         */
    Tensor(const std::vector<value_type>& data, const shape_type& shape, 
           bool requires_grad = false);

        /**
         * @brief Copy-constructs a tensor.
         * @param other Source tensor to copy.
         */
    Tensor(const Tensor& other);

        /**
         * @brief Move-constructs a tensor.
         * @param other Source tensor to move.
         */
    Tensor(Tensor&& other) noexcept;

        /**
         * @brief Copy-assigns a tensor.
         * @param other Source tensor to copy.
         * @return Reference to this tensor.
         */
    Tensor& operator=(const Tensor& other);

        /**
         * @brief Move-assigns a tensor.
         * @param other Source tensor to move.
         * @return Reference to this tensor.
         */
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor() = default;
    
    /**
     * @brief Returns the tensor shape.
     * @return Shape vector describing tensor dimensions.
     */
    const shape_type& shape() const { return shape_; }

    /**
     * @brief Returns the number of dimensions.
     * @return Rank of the tensor.
     */
    size_t ndim() const { return shape_.size(); }

    /**
     * @brief Returns total number of stored elements.
     * @return Element count in the flat storage buffer.
     */
    size_t size() const;

    /**
     * @brief Returns mutable pointer to contiguous storage.
     * @return Mutable raw pointer to element data.
     */
    value_type* data() { return data_.data(); }

    /**
     * @brief Returns read-only pointer to contiguous storage.
     * @return Const raw pointer to element data.
     */
    const value_type* data() const { return data_.data(); }

    /**
     * @brief Reports whether gradient tracking is enabled.
     * @return True when this tensor participates in autograd.
     */
    bool requires_grad() const { return requires_grad_; }

    /**
     * @brief Enables or disables gradient tracking.
     * @param value True to enable autograd recording, false to disable.
     */
    void set_requires_grad(bool value) { requires_grad_ = value; }
    
    /**
     * @brief Returns mutable element access by flat index.
     * @param idx Zero-based flat index.
     * @return Mutable reference to element at idx.
     */
    value_type& operator[](size_t idx) { return data_[idx]; }

    /**
     * @brief Returns read-only element access by flat index.
     * @param idx Zero-based flat index.
     * @return Const reference to element at idx.
     */
    const value_type& operator[](size_t idx) const { return data_[idx]; }

    /**
     * @brief Returns mutable access using multi-dimensional coordinates.
     * @param indices Per-dimension coordinates.
     * @return Mutable reference to indexed element.
     * @throws std::invalid_argument If index rank does not match tensor rank.
     * @throws std::out_of_range If any coordinate is outside shape bounds.
     */
    value_type& at(const std::vector<size_t>& indices);

    /**
     * @brief Returns read-only access using multi-dimensional coordinates.
     * @param indices Per-dimension coordinates.
     * @return Const reference to indexed element.
     * @throws std::invalid_argument If index rank does not match tensor rank.
     * @throws std::out_of_range If any coordinate is outside shape bounds.
     */
    const value_type& at(const std::vector<size_t>& indices) const;
    
    /**
     * @brief Computes element-wise addition.
     * @param other Tensor added element-wise.
     * @return Sum tensor with same shape as operands.
     * @throws std::invalid_argument If tensor shapes are not equal.
     */
    Tensor operator+(const Tensor& other) const;

    /**
     * @brief Computes element-wise subtraction.
     * @param other Tensor subtracted element-wise.
     * @return Difference tensor with same shape as operands.
     * @throws std::invalid_argument If tensor shapes are not equal.
     */
    Tensor operator-(const Tensor& other) const;

    /**
     * @brief Computes element-wise multiplication.
     * @param other Tensor multiplied element-wise.
     * @return Product tensor with same shape as operands.
     * @throws std::invalid_argument If tensor shapes are not equal.
     */
    Tensor operator*(const Tensor& other) const;

    /**
     * @brief Computes element-wise division.
     * @param other Tensor used as element-wise divisor.
     * @return Quotient tensor with same shape as operands.
     * @throws std::invalid_argument If tensor shapes are not equal.
     * @throws std::runtime_error If any divisor element is zero.
     */
    Tensor operator/(const Tensor& other) const;

    /**
     * @brief Adds scalar to all tensor elements.
     * @param scalar Scalar value to add.
     * @return Tensor with scalar-added values.
     */
    Tensor operator+(value_type scalar) const;

    /**
     * @brief Subtracts scalar from all tensor elements.
     * @param scalar Scalar value to subtract.
     * @return Tensor with scalar-subtracted values.
     */
    Tensor operator-(value_type scalar) const;

    /**
     * @brief Multiplies all tensor elements by scalar.
     * @param scalar Scalar multiplier.
     * @return Tensor with scaled values.
     */
    Tensor operator*(value_type scalar) const;

    /**
     * @brief Divides all tensor elements by scalar.
     * @param scalar Scalar divisor.
     * @return Tensor with scaled values.
     * @throws std::runtime_error If scalar is zero.
     */
    Tensor operator/(value_type scalar) const;

    /**
     * @brief Adds another tensor in place.
     * @param other Tensor added element-wise.
     * @return Reference to this tensor.
     * @throws std::invalid_argument If tensor shapes are not equal.
     */
    Tensor& operator+=(const Tensor& other);

    /**
     * @brief Subtracts another tensor in place.
     * @param other Tensor subtracted element-wise.
     * @return Reference to this tensor.
     * @throws std::invalid_argument If tensor shapes are not equal.
     */
    Tensor& operator-=(const Tensor& other);

    /**
     * @brief Multiplies by another tensor in place.
     * @param other Tensor multiplied element-wise.
     * @return Reference to this tensor.
     * @throws std::invalid_argument If tensor shapes are not equal.
     */
    Tensor& operator*=(const Tensor& other);

    /**
     * @brief Multiplies by scalar in place.
     * @param scalar Scalar multiplier.
     * @return Reference to this tensor.
     */
    Tensor& operator*=(value_type scalar);
    
    /**
     * @brief Performs 2D matrix multiplication.
     * @param other Right-hand matrix.
     * @return Matrix product tensor.
     * @throws std::invalid_argument If either tensor is not 2D or dimensions are incompatible.
     */
    Tensor matmul(const Tensor& other) const;

    /**
     * @brief Returns matrix transpose for a 2D tensor.
     * @return Transposed 2D tensor.
     * @throws std::invalid_argument If tensor is not 2D.
     */
    Tensor transpose() const;
    
    /**
     * @brief Reduces all elements by summation.
     * @return Rank-1 tensor of shape {1} containing total sum.
     */
    Tensor sum() const;

    /**
     * @brief Reduces elements by summation along one axis.
     * @param axis Reduction axis. Negative values are normalized from the end.
     * @return Tensor with axis removed and remaining dimensions preserved.
     * @throws std::invalid_argument If reducing axis on empty shape.
     * @throws std::out_of_range If axis is invalid.
     */
    Tensor sum(int axis) const;

    /**
     * @brief Computes mean across all elements.
     * @return Rank-1 tensor of shape {1} containing global mean.
     */
    Tensor mean() const;

    /**
     * @brief Computes mean along one axis.
     * @param axis Reduction axis. Negative values are normalized from the end.
     * @return Tensor with axis removed and averaged values.
     * @throws std::invalid_argument If reducing axis on empty shape.
     * @throws std::out_of_range If axis is invalid.
     */
    Tensor mean(int axis) const;

    /**
     * @brief Returns largest stored value.
     * @return Maximum element value.
     */
    value_type max() const;

    /**
     * @brief Returns smallest stored value.
     * @return Minimum element value.
     */
    value_type min() const;
    
    /**
     * @brief Returns tensor data reinterpreted with a new shape.
     * @param new_shape Requested output shape.
     * @return Tensor containing same element order with new shape.
     * @throws std::invalid_argument If new_shape product differs from size().
     */
    Tensor reshape(const shape_type& new_shape) const;

    /**
     * @brief Flattens tensor into one dimension.
     * @return Tensor with shape {size()}.
     */
    Tensor flatten() const;

    /**
     * @brief Returns a slice range along selected axis.
     * @param start Inclusive start index on axis.
     * @param end Exclusive end index on axis.
     * @param axis Axis to slice, currently only axis 0 is supported.
     * @return Sliced tensor containing copied values.
     * @throws std::out_of_range If axis is invalid.
     * @throws std::invalid_argument If start/end range is invalid.
     * @throws std::runtime_error If axis other than 0 is requested.
     */
    Tensor slice(size_t start, size_t end, size_t axis = 0) const;
    
    /**
     * @brief Creates zero-filled tensor.
     * @param shape Tensor dimensions.
     * @param requires_grad Whether returned tensor should track gradients.
     * @return Zero-initialized tensor.
     */
    static Tensor zeros(const shape_type& shape, bool requires_grad = false);

    /**
     * @brief Creates one-filled tensor.
     * @param shape Tensor dimensions.
     * @param requires_grad Whether returned tensor should track gradients.
     * @return One-initialized tensor.
     */
    static Tensor ones(const shape_type& shape, bool requires_grad = false);

    /**
     * @brief Creates tensor sampled from standard normal distribution.
     * @param shape Tensor dimensions.
     * @param requires_grad Whether returned tensor should track gradients.
     * @return Random tensor sampled from N(0,1).
     */
    static Tensor randn(const shape_type& shape, bool requires_grad = false);

    /**
     * @brief Creates tensor sampled from uniform distribution.
     * @param shape Tensor dimensions.
     * @param low Lower sampling bound.
     * @param high Upper sampling bound.
     * @param requires_grad Whether returned tensor should track gradients.
     * @return Random tensor sampled from uniform distribution.
     */
    static Tensor uniform(const shape_type& shape, value_type low = 0.0, 
                         value_type high = 1.0, bool requires_grad = false);
    
    /**
     * @brief Returns mutable gradient tensor.
     * @return Mutable gradient tensor reference.
     */
    Tensor& grad() { return *grad_; }

    /**
     * @brief Returns read-only gradient tensor.
     * @return Const gradient tensor reference.
     */
    const Tensor& grad() const { return *grad_; }

    /**
     * @brief Indicates whether gradient storage exists.
     * @return True when gradient tensor has been allocated.
     */
    bool has_grad() const { return grad_ != nullptr; }

    /**
     * @brief Returns shared pointer to gradient storage.
     * @return Shared gradient tensor pointer.
     */
    std::shared_ptr<Tensor> grad_ptr() const { return grad_; }

    /** @brief Allocates gradient tensor when not already present. */
    void ensure_grad();

    /**
     * @brief Stores local backward callback and parent tensor list.
     * @param backward_fn Callback implementing local derivative propagation.
     * @param parents Upstream tensors used during graph traversal.
     */
    void set_autograd(std::function<void()> backward_fn,
                      std::vector<std::shared_ptr<Tensor>> parents);

    /**
     * @brief Creates shared tensor copy for autograd parent wiring.
     * @param tensor Source tensor to alias.
     * @return Shared pointer holding copied tensor state.
     */
    static std::shared_ptr<Tensor> alias(const Tensor& tensor);

    /**
     * @brief Starts backpropagation from this tensor.
     * @throws std::runtime_error If requires_grad() is false.
     */
    void backward();

    /** @brief Resets gradient values to zero when gradient storage exists. */
    void zero_grad();
    
    /** @brief Prints compact tensor summary to standard output. */
    void print() const;

    /**
     * @brief Streams compact tensor representation.
     * @param os Output stream.
     * @param t Tensor to serialize.
     * @return Reference to output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t);
    
private:
    /**
     * @brief Converts multi-dimensional coordinates into flat row-major index.
     * @param indices Per-dimension indices.
     * @return Flat buffer index.
     */
    size_t compute_index(const std::vector<size_t>& indices) const;

    /**
     * @brief Validates shape compatibility for element-wise operations.
     * @param other Other tensor participating in operation.
     * @throws std::invalid_argument If shape() differs from other.shape().
     */
    void check_shape_compatible(const Tensor& other) const;

    /** @brief Allocates zero-initialized gradient tensor matching shape(). */
    void allocate_grad();
};

} // namespace nn
