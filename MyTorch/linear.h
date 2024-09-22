#ifndef LINEAR_H
#define LINEAR_H

#include <vector>
#include <memory>
#include <random>
#include "common.h"

template <typename T>
class Linear : public Module<T> {
public:
    // 构造函数：初始化 in_features 和 out_features
    Linear(int in_features, int out_features, bool bias = true) 
        : in_features(in_features), out_features(out_features) {
        
        // 初始化权重参数，形状为 (out_features, in_features)
        weight = std::make_shared<Tensor<T>>(std::vector<T>(out_features * in_features), 
                                             std::vector<size_t>{out_features, in_features}, 
                                             true, false);

        // 可选的偏置参数，形状为 (out_features)
        if (bias) {
            bias_ = std::make_shared<Tensor<T>>(std::vector<T>(out_features), 
                                                std::vector<size_t>{out_features}, 
                                                true, false);
        } else {
            bias_ = nullptr;
        }

        // 注册成员到 Module 中
        this->register_member(weight);
        if (bias) {
            this->register_member(bias_);
        }

        // 初始化参数
        reset_parameters();
    }

    // 重置参数
    void reset_parameters() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<T> dist(0, 1);

        // 初始化权重为标准正态分布
        for (auto& w : weight->data) {
            w = dist(gen);
        }

        // 如果有偏置，初始化为0
        if (bias_) {
            for (auto& b : bias_->data) {
                b = static_cast<T>(0);
            }
        }
    }

    // 前向传播
    std::shared_ptr<Tensor<T>> forward(const std::shared_ptr<Tensor<T>>& input) override {
        // 确保输入的最后一个维度等于 in_features
        assert(input->shape.back() == in_features && "Input size does not match in_features");

        // 将 input 张量 reshape 以便进行矩阵乘法
        auto input_reshaped = input->reshape({-1, in_features}); // 将形状变为 (batch_size, in_features)

        // 使用 Function 类中的矩阵乘法实现前向传播
        auto matmul_func = std::make_shared<MatMul<T>>();
        auto x1 = matmul_func->launch({input_reshaped, weight->transpose()}); // 矩阵乘法

        // 如果有偏置，添加偏置
        if (bias_) {
            // 将偏置进行广播，使其能够添加到 x 的每一行
            auto broadcast_func = std::make_shared<Broadcast<T>>();
            auto bias_broadcasted = broadcast_func->launch({bias_, x1});
            auto add_fn = std::make_shared<Add<T>>();
            x = add_fn->launch({x1, bias_broadcasted});
        }

        // 将结果 reshape 回原始输入的形状，除了最后一维是 out_features
        std::vector<size_t> output_shape = input->shape;
        output_shape.back() = out_features; // 更新最后一维为 out_features
        return x->reshape(output_shape);
    }

private:
    int in_features, out_features;
    std::shared_ptr<Tensor<T>> weight;
    std::shared_ptr<Tensor<T>> bias_;
};

#endif // LINEAR_H
