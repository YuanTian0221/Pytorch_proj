#ifndef SGD_OPTIMIZER_H
#define SGD_OPTIMIZER_H

#include <vector>
#include <memory>
#include "tensor.h" // 假设你已经有了 Tensor 类的定义
#include "optimizer.h" // 假设你已经有了 Optimizer 基类的定义

// SGD 优化器类
template <typename T>
class SGD : public Optimizer<T> {
public:
    // 构造函数
    SGD(const std::vector<std::shared_ptr<Tensor<T>>>& params, T lr);

    // 实现 step() 函数
    void step() override;

private:
    T lr;  // 学习率
};

// 实现构造函数
template <typename T>
SGD<T>::SGD(const std::vector<std::shared_ptr<Tensor<T>>>& params, T lr)
    : Optimizer<T>(params), lr(lr) {}

// 实现 step() 函数
template <typename T>
void SGD<T>::step() {
    for (auto& param : this->params) {
        if (param->grad.empty()) continue;
        for (size_t i = 0; i < param->data.size(); ++i) {
            param->data[i] -= lr * param->grad[i];
        }
    }
}

#endif // SGD_OPTIMIZER_H
