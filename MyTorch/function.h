#ifndef FUNCTION_H
#define FUNCTION_H

#include <vector>
#include <memory>
#include <algorithm>
#include <cassert>
#include <iostream>

// 前向声明 Tensor 类
template <typename T>
class Tensor;

template <typename T>
class Function : public std::enable_shared_from_this<Function<T>> {
public:
    std::vector<std::shared_ptr<Tensor<T>>> saved_tensors;  // 保存前向传播所需的张量
    std::vector<std::shared_ptr<Tensor<T>>> inputs;         // 保存输入张量，方便反向传播
    std::vector<std::weak_ptr<Tensor<T>>> outputs;          // 输出张量的弱引用，避免内存泄漏
    int generation = 0;  // 用于控制反向传播顺序

    void save_for_backward(const std::vector<std::shared_ptr<Tensor<T>>>& tensors);
    void set_inputs(const std::vector<std::shared_ptr<Tensor<T>>>& inputs_);

    virtual std::shared_ptr<Tensor<T>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) = 0;
    virtual std::vector<std::shared_ptr<Tensor<T>>> backward(const std::shared_ptr<Tensor<T>>& grad_output) = 0;

    std::shared_ptr<Tensor<T>> launch(const std::vector<std::shared_ptr<Tensor<T>>>& xs);
};

// 包含实现文件
#include "function_impl.h"

#endif  // FUNCTION_H