#include "tensor.h"
#include "tensor_impl.h"

// 将模板类函数的实现放在 tensor_impl.h 中
// tensor.cpp用于实现非模板类的成员函数。
// 广播机制辅助函数(不是模板类的函数)
std::vector<size_t> broadcast_shapes(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2) {
    std::vector<size_t> result;
    auto it1 = shape1.rbegin();
    auto it2 = shape2.rbegin();
    while (it1 != shape1.rend() || it2 != shape2.rend()) {
        size_t dim1 = (it1 != shape1.rend()) ? *it1++ : 1;
        size_t dim2 = (it2 != shape2.rend()) ? *it2++ : 1;
        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            throw std::invalid_argument("Shapes cannot be broadcasted");
        }
        result.push_back(std::max(dim1, dim2));
    }
    std::reverse(result.begin(), result.end());
    return result;
}

