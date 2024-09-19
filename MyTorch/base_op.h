#ifndef BASE_OP_H
#define BASE_OP_H
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>
#include "tensor.h"
#include "function.h"


// Add 类，继承自 Function，支持多个输入
template <typename T>
class Add : public Function<T> {
public:
    std::vector<size_t> out_shape;
    // 前向传播：计算多个 Tensor 的加法
    std::shared_ptr<Tensor<T>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override {
        this->inputs = inputs;  // 保存输入
        std::vector<T> result(inputs[0]->data.size(), static_cast<T>(0)); // 初始化结果
        
        // 逐元素累加
        for (size_t i = 0; i < inputs.size(); ++i) {
            for (size_t j = 0; j < result.size(); ++j) {
                result[j] += inputs[i]->data[j];
            }
        }
        // 计算输出tensor的shape
        out_shape = inputs[0]->shape;

        // 创建输出 Tensor
        auto output = std::make_shared<Tensor<T>>(result, out_shape, true, false);  // 输出 Tensor 需要计算梯度
        //output->set_creator(this->shared_from_this());  // 设置输出的 creator
        //std::cout << "Creator generation: " << output->creator->generation << std::endl;
        //std::cout << "Result generation: " << output->generation << std::endl;
        
        return output;
    }

    // 反向传播：加法的反向传播，梯度不变，所有输入的梯度都是传入的梯度
    std::vector<std::shared_ptr<Tensor<T>>> backward(const std::shared_ptr<Tensor<T>>& grad_output) override {
        std::vector<std::shared_ptr<Tensor<T>>> grad_inputs;
        
        // 对每个输入，返回相同的 grad_output
        for (size_t i = 0; i < this->inputs.size(); ++i) {
            auto grad_input = std::make_shared<Tensor<T>>(grad_output->data, grad_output->shape, false, grad_output->device);
            grad_inputs.push_back(grad_input);
        }
        
        return grad_inputs;
    }
};

#endif
