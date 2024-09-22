#include "common.h"
#include "mse_loss.h"


int main() {
    // 创建输入和目标张量
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f};
    std::vector<float> target_data = {1.0f, 2.0f, 4.0f};
    std::vector<size_t> shape = {3};  // 一维张量

    auto input = std::make_shared<Tensor<float>>(input_data, shape, true, false);
    auto target = std::make_shared<Tensor<float>>(target_data, shape, true, false);

    // 创建 MSELoss 实例
    MSELoss<float> mse_loss("mean");

    // 计算损失
    auto loss = mse_loss.forward({input, target});

    // 输出结果
    std::cout << "MSE Loss: " << loss->data[0] << std::endl;  // 期望输出 0.3333

    return 0;

}