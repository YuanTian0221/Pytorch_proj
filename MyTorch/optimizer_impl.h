#ifndef OPTIMIZER_IMPL_H
#define OPTIMIZER_IMPL_H

#include "optimizer.h"

// 构造函数实现
template <typename T>
Optimizer<T>::Optimizer(const std::vector<std::shared_ptr<Tensor<T>>>& params)
    : params(params) {}

// zero_grad 函数实现
template <typename T>
void Optimizer<T>::zero_grad() {
    for (auto& param : params) {
        param->zero_grad();
    }
}

#endif  // OPTIMIZER_IMPL_H
