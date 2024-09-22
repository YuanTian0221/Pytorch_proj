#ifndef MSE_LOSS_H
#define MSE_LOSS_H

#include "loss.h"  // 假设你有一个基类 _Loss 的定义
#include "loss_impl.h"  // 假设你有一个基类 _Loss 的定义
#include "tensor.h"  // 假设你有一个 Tensor 类

template <typename T>
class MSELoss : public _Loss<T> {
public:
    MSELoss(const std::string& reduction = "mean") : _Loss<T>(reduction) {}

protected:
    std::shared_ptr<Tensor<T>> compute_loss(const std::shared_ptr<Tensor<T>>& input,
                                            const std::shared_ptr<Tensor<T>>& target) override {
        // 计算平方误差 (input - target) ** 2
        std::vector<T> errors(input->data.size());
        for (size_t i = 0; i < input->data.size(); ++i) {
            errors[i] = (input->data[i] - target->data[i]) * (input->data[i] - target->data[i]);
        }

        // 根据 reduction 模式计算损失
        std::shared_ptr<Tensor<T>> loss;
        if (this->reduction == "mean") {
            T sum = std::accumulate(errors.begin(), errors.end(), static_cast<T>(0));
            loss = std::make_shared<Tensor<T>>(sum / static_cast<T>(input->data.size()), false, false);
        } else if (this->reduction == "sum") {
            T sum = std::accumulate(errors.begin(), errors.end(), static_cast<T>(0));
            loss = std::make_shared<Tensor<T>>(sum, false, false);
        } else {
            loss = std::make_shared<Tensor<T>>(errors, input->shape, false, input->device);
        }

        return loss;
    }
};

#endif  // MSE_LOSS_H
