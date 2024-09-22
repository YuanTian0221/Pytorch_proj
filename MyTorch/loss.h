#ifndef LOSS_H
#define LOSS_H

#include <string>
#include "module.h"
#include "module_impl.h"

template <typename T>
class _Loss : public Module<T> {
public:
    std::string reduction; // none | mean | sum

    // 构造函数
    _Loss(const std::string& reduction = "mean");

    // 实现具有两个参数的 forward 函数
    std::shared_ptr<Tensor<T>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override {
        if (inputs.size() != 2) {
            throw std::invalid_argument("Loss requires exactly two inputs.");
        }
        return compute_loss(inputs[0], inputs[1]);
    }

    // 在基类中实现 __call__ 的等价函数
    std::shared_ptr<Tensor<T>> operator()(const std::shared_ptr<Tensor<T>>& input, const std::shared_ptr<Tensor<T>>& target);

    // 用于返回reduction方式
    std::string get_reduction() const;

    // 设置reduction方式
    void set_reduction(const std::string& reduction_type);

protected:
    // 子类将实现此方法来计算损失
    virtual std::shared_ptr<Tensor<T>> compute_loss(const std::shared_ptr<Tensor<T>>& input,
                                                    const std::shared_ptr<Tensor<T>>& target) = 0;
};

#include "loss_impl.h"  // 包含实现文件

#endif // LOSS_H
