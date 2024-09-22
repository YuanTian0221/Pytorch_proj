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
    // 创建 Add 函数
    auto Add_func1 = std::make_shared<Add<float>>();
    auto Add_func2 = std::make_shared<Add<float>>();

    // 前向传播
    auto result = Add_func1->launch({tensor1, tensor2});
    auto result2 = Add_func2->launch({tensor1, result});

    // 打印前向传播结果
    std::cout << "Forward result: ";
    for (auto v : result2->data) {
        std::cout << v << " ";  // 结果应该是 {5.0, 7.0, 9.0}
    }
    std::cout << std::endl;
    // 反向传播
    result2->backward();

    // 打印梯度
    std::cout << "Result2 creator: "<< result2->creator->generation << std::endl;
    std::cout << "Generation for result2: " << result2->generation << std::endl;
    for (auto v : result2->grad) {
        std::cout << v << " ";  // 结果应该是 {1.0, 1.0, 1.0}
    }
    std::cout << std::endl;

    std::cout << "Result Creator: "<< result->creator->generation << std::endl;
    std::cout << "Generation for result: " << result->generation << std::endl;
    for (auto v : result->grad) {
        std::cout << v << " ";  // 结果应该是 {1.0, 1.0, 1.0}
    }
    std::cout << std::endl;

    //std::cout << "Tensor1 Creator: "<< tensor1->creator->generation << std::endl;
    std::cout << "Generation for tensor1: " << tensor1->generation << std::endl;
    for (auto v : tensor1->grad) {
        std::cout << v << " ";  // 结果应该是 {1.0, 1.0, 1.0}
    }
    std::cout << std::endl;

    //std::cout << "Tensor2 Creator: "<< tensor2->creator->generation << std::endl;
    std::cout << "Generation for tensor2: " << tensor2->generation << std::endl;
    for (auto v : tensor2->grad) {
        std::cout << v << " ";  // 结果应该是 {1.0, 1.0, 1.0}
    }
    std::cout << std::endl;

    //std::cout << "Tensor3 Creator: "<< tensor3->creator->generation << std::endl;
    std::cout << "Generation for tensor3: " << tensor3->generation << std::endl;
    for (auto v : tensor3->grad) {
        std::cout << v << " ";  // 结果应该是 {1.0, 1.0, 1.0}
    }
    std::cout << std::endl;

    return 0;
}