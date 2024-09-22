#include <iostream>
#include <vector>
#include <memory>
#include "common.h" // 假设你已经有了 Tensor 类的定义
#include "sgd_optimizer.h" // 假设你已经有了 SGD 类的定义

int main() {
    // 创建一个 Tensor 实例
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    std::vector<float> grad = {0.1f, 0.1f, 0.1f};  // 假设梯度都为 0.1
    std::vector<size_t> shape = {3};

    auto tensor = std::make_shared<Tensor<float>>(data, shape, true, false);  // 需要计算梯度
    tensor->grad = grad;  // 设置梯度

    // 创建 SGD 优化器实例，学习率为 0.01
    float learning_rate = 0.01f;
    std::vector<std::shared_ptr<Tensor<float>>> params = {tensor};
    SGD<float> optimizer(params, learning_rate);

    // 调用 step() 函数进行优化
    optimizer.step();

    // 打印更新后的 Tensor 数据
    std::cout << "Updated Tensor data: ";
    for (auto v : tensor->data) {
        std::cout << v << " ";  // 结果应该是 {1.0 - 0.01*0.1, 2.0 - 0.01*0.1, 3.0 - 0.01*0.1}
    }
    std::cout << std::endl;

    return 0;
}
