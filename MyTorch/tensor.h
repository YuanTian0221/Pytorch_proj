#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <numeric>
#include <algorithm>
#include <initializer_list>
#include <cassert>
#include <set>

template <typename T>
class Tensor : public std::enable_shared_from_this<Tensor<T>> {
public:
    // 数据成员
    std::vector<T> data;  // 存储张量数据
    std::vector<size_t> shape;  // 张量的形状
    std::vector<size_t> strides;  // 用于计算一维索引的步幅
    std::vector<T> grad;  // 梯度，与 data 大小相同
    bool requires_grad;  // 是否需要计算梯度

    // 自动微分相关
    std::function<void()> _backward;  // 反向传播函数
    std::vector<std::shared_ptr<Tensor<T>>> _prev;  // 前驱节点
    std::string _op;  // 生成该张量的操作符

    // 构造函数
    Tensor(const std::vector<T>& data,
           const std::vector<size_t>& shape,
           bool requires_grad = false,
           const std::vector<std::shared_ptr<Tensor<T>>>& children = {},
           const std::string& op = "");
    
    // 支持从标量构造
    Tensor(T value, bool requires_grad = false);

    // 元素访问
    T& operator[](const std::vector<size_t>& indices);
    const T& operator[](const std::vector<size_t>& indices) const;

    // 基本算术运算
    std::shared_ptr<Tensor<T>> operator+(const std::shared_ptr<Tensor<T>>& other);
    std::shared_ptr<Tensor<T>> operator-(const std::shared_ptr<Tensor<T>>& other);
    std::shared_ptr<Tensor<T>> operator*(const std::shared_ptr<Tensor<T>>& other);
    std::shared_ptr<Tensor<T>> operator/(const std::shared_ptr<Tensor<T>>& other);
    std::shared_ptr<Tensor<T>> matmul(const std::shared_ptr<Tensor<T>>& other);
    // 反向传播
    void backward();

    // 辅助函数
    void zero_grad();  // 清零梯度
    void reshape(const std::vector<size_t>& new_shape);  // 重塑张量形状
    void print() const;  // 打印张量信息
};

// 加法运算符重载
template <typename T>
std::shared_ptr<Tensor<T>> operator+(const std::shared_ptr<Tensor<T>>& lhs, const std::shared_ptr<Tensor<T>>& rhs) {
    return lhs->operator+(rhs);  // 调用 Tensor 类内部的 operator+
}
// 减法运算符重载
template <typename T>
std::shared_ptr<Tensor<T>> operator-(const std::shared_ptr<Tensor<T>>& lhs, const std::shared_ptr<Tensor<T>>& rhs) {
    return lhs->operator-(rhs);  // 调用 Tensor 类内部的 operator+
}
// 乘法运算符重载
template <typename T>
std::shared_ptr<Tensor<T>> operator*(const std::shared_ptr<Tensor<T>>& lhs, const std::shared_ptr<Tensor<T>>& rhs) {
    return lhs->operator*(rhs);  // 调用 Tensor 类内部的 operator*
}
// 除法运算符重载
template <typename T>
std::shared_ptr<Tensor<T>> operator/(const std::shared_ptr<Tensor<T>>& lhs, const std::shared_ptr<Tensor<T>>& rhs) {
    return lhs->operator/(rhs);  // 调用 Tensor 类内部的 operator*
}
// 声明外部广播函数
std::vector<size_t> broadcast_shapes(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2);

#endif