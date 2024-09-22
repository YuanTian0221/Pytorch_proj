#include "common.h"

int main() {

    // 创建两个 Tensor
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 3.0f};
    std::vector<float> data2 = {4.0f, 5.0f, 6.0f, 6.0f};
    std::vector<float> data3 = {7.0f, 8.0f, 9.0f, 9.0f};
    std::vector<size_t> shape = {2, 2};

    auto tensor1 = std::make_shared<Tensor<float>>(data1, shape, true);  // 需要计算梯度
    auto tensor2 = std::make_shared<Tensor<float>>(data2, shape, true);  // 需要计算梯度
    auto tensor3 = std::make_shared<Tensor<float>>(data3, shape, true);  // 需要计算梯度
    
    tensor1->transpose();
    // 打印前向传播结果
    std::cout << "Forward result: ";
    for (auto v : tensor1->data) {
        std::cout << v << " ";  // 结果应该是 {5.0, 7.0, 9.0}
    }
    std::cout << std::endl;

    return 0;
}