#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <memory>
#include "tensor.h"  // 假设你已经定义了 Tensor 类

template <typename T>
class Optimizer {
public:
    // 构造函数：接受一组需要优化的参数
    Optimizer(const std::vector<std::shared_ptr<Tensor<T>>>& params);

    // 清除所有参数的梯度
    void zero_grad();

    // 虚函数：执行优化步骤，子类必须实现
    virtual void step() = 0;

protected:
    std::vector<std::shared_ptr<Tensor<T>>> params;  // 保存所有需要优化的参数
};

#include "optimizer_impl.h"

#endif  // OPTIMIZER_H
