#ifndef FUNCTION_IMPL_H
#define FUNCTION_IMPL_H

#include "function.h"

template <typename T>
void Function<T>::save_for_backward(const std::vector<std::shared_ptr<Tensor<T>>>& tensors) {
    saved_tensors = tensors;
}

template <typename T>
void Function<T>::set_inputs(const std::vector<std::shared_ptr<Tensor<T>>>& inputs_) {
    inputs = inputs_;
}

template <typename T>
std::shared_ptr<Tensor<T>> Function<T>::launch(const std::vector<std::shared_ptr<Tensor<T>>>& xs) {
    // 1. 设置输入 Tensor，并执行前向传播
    set_inputs(xs);
    auto result = forward(xs);
    // 2. 检查是否需要计算梯度（requires_grad）
    bool requires_grad = std::any_of(xs.begin(), xs.end(),
        [](const std::shared_ptr<Tensor<T>>& t) {
            return t->requires_grad;
        });
    // 3. 如果需要计算梯度，设置生成关系
    if (requires_grad) {
        // 记录生成的 Tensor 的 weak_ptr 防止循环引用
        outputs.push_back(result);
        // 设置生成顺序（反向传播的顺序）
        generation = std::max_element(xs.begin(), xs.end(),
            [](const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b) {
                return a->generation < b->generation;
            })->get()->generation + 1;
        // 设置输出 Tensor 的 generation 为当前 Function 的 generation
        result->generation = generation;
        // 设置输出 Tensor 的生成者
        result->set_creator(this->shared_from_this());
        //std::cout << "Creator generation: " << result->creator->generation << std::endl;
        //std::cout << "Result generation: " << result->generation << std::endl;
        
    }

    return result;
}

#endif  // FUNCTION_IMPL_H
