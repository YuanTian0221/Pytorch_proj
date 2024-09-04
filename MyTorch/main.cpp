#include "tensor.h"
#include "tensor_impl.h"

int main() {
    // 创建张量 a 和 b
    auto a = std::make_shared<Tensor<float>>(std::vector<float>{1, 2, 3, 4}, std::vector<size_t>{2, 2}, true);
    auto b = std::make_shared<Tensor<float>>(std::vector<float>{5, 6, 7, 8}, std::vector<size_t>{2, 2}, true);
    
    // 进行计算
    //auto c = a - b;  // 加法
    //auto d = a / b;  // 乘法
    //auto e = c / d;  // 组合操作
    auto e = a->matmul(b);
    // 前向结果
    e->print();
    
    // 反向传播
    e->backward();
    
    // 打印梯度
    std::cout << "Gradient of a:" << std::endl;
    for (auto val : a->grad) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Gradient of b:" << std::endl;
    for (auto val : b->grad) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
