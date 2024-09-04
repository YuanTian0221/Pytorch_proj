#ifndef TENSOR_IMPL_H
#define TENSOR_IMPL_H

#include "tensor.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <stdexcept>

// Tensor 类构造函数实现
template <typename T>
Tensor<T>::Tensor(const std::vector<T>& data, const std::vector<size_t>& shape, bool requires_grad,
                  const std::vector<std::shared_ptr<Tensor<T>>>& children, const std::string& op)
    : data(data), shape(shape), requires_grad(requires_grad), _prev(children), _op(op) {
    strides.resize(shape.size());
    size_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    if (requires_grad) {
        grad = std::vector<T>(data.size(), static_cast<T>(0));
    }
}

template <typename T>
Tensor<T>::Tensor(T value, bool requires_grad)
    : data({value}), shape({1}), requires_grad(requires_grad) {
    strides = {1};
    if (requires_grad) {
        grad = std::vector<T>(1, static_cast<T>(0));
    }
}

// 元素访问
template <typename T>
T& Tensor<T>::operator[](const std::vector<size_t>& indices) {
    assert(indices.size() == shape.size());
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        assert(indices[i] < shape[i]);
        offset += strides[i] * indices[i];
    }
    return data[offset];
}

template <typename T>
const T& Tensor<T>::operator[](const std::vector<size_t>& indices) const {
    assert(indices.size() == shape.size());
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        assert(indices[i] < shape[i]);
        offset += strides[i] * indices[i];
    }
    return data[offset];
}

// 加法运算
template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::operator+(const std::shared_ptr<Tensor<T>>& other) {
    std::vector<size_t> result_shape = broadcast_shapes(this->shape, other->shape);
    size_t total_size = std::accumulate(result_shape.begin(), result_shape.end(), 1, std::multiplies<size_t>());
    std::vector<T> result_data(total_size);
    
    for (size_t i = 0; i < total_size; ++i) {
        std::vector<size_t> idx(result_shape.size());
        size_t temp = i;
        for (int j = result_shape.size() - 1; j >= 0; --j) {
            idx[j] = temp % result_shape[j];
            temp /= result_shape[j];
        }
        T a = this->data.size() == 1 ? this->data[0] : (*this)[idx];
        T b = other->data.size() == 1 ? other->data[0] : (*other)[idx];
        result_data[i] = a + b;
    }
    
    auto out = std::make_shared<Tensor<T>>(result_data, result_shape, this->requires_grad || other->requires_grad,
                                           std::vector<std::shared_ptr<Tensor<T>>>{this->shared_from_this(), other}, "+");
    
    out->_backward = [this, other, out]() {
        if (this->requires_grad) {
            for (size_t i = 0; i < this->data.size(); ++i) {
                this->grad[i] += out->grad[i];
            }
        }
        if (other->requires_grad) {
            for (size_t i = 0; i < other->data.size(); ++i) {
                other->grad[i] += out->grad[i];
            }
        }
    };
    
    return out;
}

// 减法运算
template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::operator-(const std::shared_ptr<Tensor<T>>& other) {
    std::vector<size_t> result_shape = broadcast_shapes(this->shape, other->shape);
    size_t total_size = std::accumulate(result_shape.begin(), result_shape.end(), 1, std::multiplies<size_t>());
    std::vector<T> result_data(total_size);
    
    for (size_t i = 0; i < total_size; ++i) {
        std::vector<size_t> idx(result_shape.size());
        size_t temp = i;
        for (int j = result_shape.size() - 1; j >= 0; --j) {
            idx[j] = temp % result_shape[j];
            temp /= result_shape[j];
        }
        T a = this->data.size() == 1 ? this->data[0] : (*this)[idx];
        T b = other->data.size() == 1 ? other->data[0] : (*other)[idx];
        result_data[i] = a - b;
    }
    
    auto out = std::make_shared<Tensor<T>>(result_data, result_shape, this->requires_grad || other->requires_grad,
                                           std::vector<std::shared_ptr<Tensor<T>>>{this->shared_from_this(), other}, "-");
    
    out->_backward = [this, other, out]() {
        if (this->requires_grad) {
            for (size_t i = 0; i < this->data.size(); ++i) {
                this->grad[i] += out->grad[i];
            }
        }
        if (other->requires_grad) {
            for (size_t i = 0; i < other->data.size(); ++i) {
                other->grad[i] -= out->grad[i];
            }
        }
    };
    
    return out;
}

// 乘法运算
template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::operator*(const std::shared_ptr<Tensor<T>>& other) {
    std::vector<size_t> result_shape = broadcast_shapes(this->shape, other->shape);
    size_t total_size = std::accumulate(result_shape.begin(), result_shape.end(), 1, std::multiplies<size_t>());
    std::vector<T> result_data(total_size);
    
    for (size_t i = 0; i < total_size; ++i) {
        std::vector<size_t> idx(result_shape.size());
        size_t temp = i;
        for (int j = result_shape.size() - 1; j >= 0; --j) {
            idx[j] = temp % result_shape[j];
            temp /= result_shape[j];
        }
        T a = this->data.size() == 1 ? this->data[0] : (*this)[idx];
        T b = other->data.size() == 1 ? other->data[0] : (*other)[idx];
        result_data[i] = a * b;
    }
    
    auto out = std::make_shared<Tensor<T>>(result_data, result_shape, this->requires_grad || other->requires_grad,
                                           std::vector<std::shared_ptr<Tensor<T>>>{this->shared_from_this(), other}, "*");
    
    out->_backward = [this, other, out]() {
        if (this->requires_grad) {
            for (size_t i = 0; i < this->data.size(); ++i) {
                this->grad[i] += other->data[i] * out->grad[i];
            }
        }
        if (other->requires_grad) {
            for (size_t i = 0; i < other->data.size(); ++i) {
                other->grad[i] += this->data[i] * out->grad[i];
            }
        }
    };
    
    return out;
}

// 除法运算
template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::operator/(const std::shared_ptr<Tensor<T>>& other) {
    std::vector<size_t> result_shape = broadcast_shapes(this->shape, other->shape);
    size_t total_size = std::accumulate(result_shape.begin(), result_shape.end(), 1, std::multiplies<size_t>());
    std::vector<T> result_data(total_size);
    
    for (size_t i = 0; i < total_size; ++i) {
        std::vector<size_t> idx(result_shape.size());
        size_t temp = i;
        for (int j = result_shape.size() - 1; j >= 0; --j) {
            idx[j] = temp % result_shape[j];
            temp /= result_shape[j];
        }
        T a = this->data.size() == 1 ? this->data[0] : (*this)[idx];
        T b = other->data.size() == 1 ? other->data[0] : (*other)[idx];
        result_data[i] = a / b;
    }
    
    auto out = std::make_shared<Tensor<T>>(result_data, result_shape, this->requires_grad || other->requires_grad,
                                           std::vector<std::shared_ptr<Tensor<T>>>{this->shared_from_this(), other}, "/");
    
    out->_backward = [this, other, out]() {
        if (this->requires_grad) {
            for (size_t i = 0; i < this->data.size(); ++i) {
                this->grad[i] += (1 / other->data[i]) * out->grad[i];
            }
        }
        if (other->requires_grad) {
            for (size_t i = 0; i < other->data.size(); ++i) {
                other->grad[i] += (-this->data[i] / (other->data[i] * other->data[i])) * out->grad[i];
            }
        }
    };
    
    return out;
}
// 矩阵相乘
template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::matmul(const std::shared_ptr<Tensor<T>>& other) {
    // 检查输入张量是否可以进行矩阵乘法
    assert(this->shape.size() == 2 && other->shape.size() == 2 && "Both tensors must be 2D.");
    assert(this->shape[1] == other->shape[0] && "Shapes are not aligned for matrix multiplication.");

    // 结果矩阵的形状
    std::vector<size_t> result_shape = {this->shape[0], other->shape[1]};
    std::vector<T> result_data(result_shape[0] * result_shape[1], static_cast<T>(0));

    // 进行矩阵乘法
    for (size_t i = 0; i < this->shape[0]; ++i) {
        for (size_t j = 0; j < other->shape[1]; ++j) {
            for (size_t k = 0; k < this->shape[1]; ++k) {
                result_data[i * result_shape[1] + j] += (*this)[{i, k}] * (*other)[{k, j}];
            }
        }
    }

    // 创建新张量并设置反向传播
    auto out = std::make_shared<Tensor<T>>(result_data, result_shape, this->requires_grad || other->requires_grad,
                                           std::vector<std::shared_ptr<Tensor<T>>>{this->shared_from_this(), other}, "matmul");

    // 定义反向传播函数
    out->_backward = [this, other, out]() {
        if (this->requires_grad) {
            // 计算对 this 的梯度：dL/dA = dL/dC * dC/dA，其中 dC/dA 是反向传播规则
            for (size_t i = 0; i < this->shape[0]; ++i) {
                for (size_t k = 0; k < this->shape[1]; ++k) {
                    for (size_t j = 0; j < other->shape[1]; ++j) {
                        this->grad[i * this->shape[1] + k] += out->grad[i * other->shape[1] + j] * (*other)[{k, j}];
                    }
                }
            }
        }
        if (other->requires_grad) {
            // 计算对 other 的梯度：dL/dB = dL/dC * dC/dB，其中 dC/dB 是反向传播规则
            for (size_t k = 0; k < this->shape[1]; ++k) {
                for (size_t j = 0; j < other->shape[1]; ++j) {
                    for (size_t i = 0; i < this->shape[0]; ++i) {
                        other->grad[k * other->shape[1] + j] += out->grad[i * other->shape[1] + j] * (*this)[{i, k}];
                    }
                }
            }
        }
    };

    return out;
}
// 反向传播
template <typename T>
void Tensor<T>::backward() {
    this->grad = std::vector<T>(this->data.size(), static_cast<T>(1));
    std::vector<std::shared_ptr<Tensor<T>>> topo;
    std::set<Tensor<T>*> visited;
    
    std::function<void(std::shared_ptr<Tensor<T>>)> build_topo = [&](std::shared_ptr<Tensor<T>> v) {
        if (visited.find(v.get()) == visited.end()) {
            visited.insert(v.get());
            for (auto child : v->_prev) {
                build_topo(child);
            }
            topo.push_back(v);
        }
    };
    
    build_topo(this->shared_from_this());
    
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->_backward) {
            (*it)->_backward();
        }
    }
}

// 辅助函数实现
template <typename T>
void Tensor<T>::zero_grad() {
    if (requires_grad) {
        std::fill(grad.begin(), grad.end(), static_cast<T>(0));
    }
}

template <typename T>
void Tensor<T>::reshape(const std::vector<size_t>& new_shape) {
    size_t total_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>());
    assert(total_size == data.size() && "New shape must have the same number of elements as the original shape.");
    shape = new_shape;
    strides.resize(shape.size());
    size_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
}

template <typename T>
void Tensor<T>::print() const {
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i != shape.size() - 1) std::cout << ", ";
    }
    std::cout << "], data=[";
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << data[i];
        if (i != data.size() - 1) std::cout << ", ";
    }
    std::cout << "])" << std::endl;
}

#endif  // TENSOR_IMPL_H
