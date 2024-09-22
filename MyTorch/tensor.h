#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <memory>
#include <initializer_list>
#include <cassert>
#include <functional>
#include <numeric>
#include <algorithm>
#include <string>
#include <set>
#include <cmath>
#include <queue>


// 前向声明 Function 类，避免循环依赖
template <typename T>
class Function;

template <typename T>
class Tensor {
public:
    // 数据成员
    std::vector<T> data;          // 存储张量数据
    std::vector<size_t> shape;    // 张量的形状
    std::vector<size_t> strides;  // 用于计算一维索引的步幅
    std::vector<T> grad;          // 梯度，与 data 大小相同
    bool requires_grad;           // 是否需要计算梯度
    bool device;                  // CpuDevice or GpuDevice
    int generation = 0;           // 表示生成张量的顺序

    // 自动微分相关
    std::shared_ptr<Function<T>> creator;  // 生成该张量的操作

    // 构造函数
    Tensor(const std::vector<T>& data,
           const std::vector<size_t>& shape,
           bool requires_grad = false,
           bool device = false);

    Tensor(T value, bool requires_grad = false, bool device = false);

    // 设置创建者
    void set_creator(const std::shared_ptr<Function<T>>& func);

    // 元素访问
    T& operator[](const std::vector<size_t>& indices);
    const T& operator[](const std::vector<size_t>& indices) const;

    // 反向传播
    void backward();

    // 打印张量
    void print() const;

    // 清零梯度
    void zero_grad();

    // 张量转置
    void transpose();
    
    // 广播
    std::shared_ptr<Tensor<T>> broadcast_to(const std::vector<size_t>& target_shape) const;

    // 广播的还原
    std::shared_ptr<Tensor<T>> sum_to(const std::vector<size_t>& target_shape) const;
};

#include "tensor_impl.h"  // 包含实现文件

#endif