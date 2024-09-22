#ifndef MODULE_IMPL_HPP
#define MODULE_IMPL_HPP

#include "module.h"

// 析构函数的实现
template <typename T>
Module<T>::~Module() = default;

// 获取所有参数的列表
template <typename T>
std::vector<std::shared_ptr<Tensor<T>>> Module<T>::parameters() {
    std::vector<std::shared_ptr<Tensor<T>>> params;
    // 遍历所有成员，寻找 Tensor 类型的成员
    for (auto& member : members_) {
        if (auto p = std::dynamic_pointer_cast<Tensor<T>>(member)) {
            params.push_back(p);
        } else if (auto m = std::dynamic_pointer_cast<Module<T>>(member)) {
            auto sub_params = m->parameters();
            params.insert(params.end(), sub_params.begin(), sub_params.end());
        }
    }
    return params;
}

// 将所有参数的梯度清零
template <typename T>
void Module<T>::zero_grad() {
    for (auto& p : parameters()) {
        p->zero_grad();
    }
}

// 用于前向传播调用
template <typename T>
std::shared_ptr<Tensor<T>> Module<T>::operator()(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) {
    return forward(input);
}

// 添加一个成员到模块中
template <typename T>
template <typename U>
void Module<T>::register_member(const std::shared_ptr<U>& member) {
    members_.push_back(member);
}


#endif // MODULE_IMPL_HPP
