#ifndef LOSS_IMPL_H
#define LOSS_IMPL_H

#include "loss.h"

// 构造函数实现
template <typename T>
_Loss<T>::_Loss(const std::string& reduction)
    : reduction(reduction) {}

// operator() 函数实现
template <typename T>
std::shared_ptr<Tensor<T>> _Loss<T>::operator()(const std::shared_ptr<Tensor<T>>& input, const std::shared_ptr<Tensor<T>>& target) {
    return forward(input, target);
}

// get_reduction 函数实现
template <typename T>
std::string _Loss<T>::get_reduction() const {
    return reduction;
}

// set_reduction 函数实现
template <typename T>
void _Loss<T>::set_reduction(const std::string& reduction_type) {
    reduction = reduction_type;
}

#endif // LOSS_IMPL_H
