#include "common.h"
#include "sgd_optimizer.h"
#include "linear.h"
#include "mse_loss.h"

int main() {
    // 定义模型：2个输入特征，1个输出
    auto model = std::make_shared<Linear<float>>(2, 1);

    // 定义优化器和损失函数
    auto optimizer = std::make_shared<SGD<float>>(model->parameters(), 1e-4f);
    auto loss_fn = std::make_shared<MSELoss<float>>("mean");

    // 数据：面积和房龄
    std::vector<float> areas = {64.4f, 68.0f, 74.1f, 74.0f, 76.9f, 78.1f, 78.6f};
    std::vector<float> ages = {31.0f, 21.0f, 19.0f, 24.0f, 17.0f, 16.0f, 17.0f};

    // 创建输入Tensor（X）：面积和房龄
    std::vector<float> X_data;
    for (size_t i = 0; i < areas.size(); ++i) {
        X_data.push_back(areas[i]);
        X_data.push_back(ages[i]);
    }
    std::vector<size_t> X_shape = {areas.size(), 2}; // 7行2列的矩阵
    auto X = std::make_shared<Tensor<float>>(X_data, X_shape, false, false);

    // 创建目标Tensor（y）：挂牌售价
    std::vector<float> prices = {6.1f, 6.25f, 7.8f, 6.66f, 7.82f, 7.14f, 8.02f};
    std::vector<size_t> y_shape = {prices.size(), 1}; // 7行1列的矩阵
    auto y = std::make_shared<Tensor<float>>(prices, y_shape, false, false);

    int epochs = 1000;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // 前向传播
        auto prediction = model->forward(X);

        // 计算损失
        auto loss = loss_fn->forward({prediction, y});

        // 梯度清零
        optimizer->zero_grad();

        // 反向传播
        loss->backward();

        // 更新参数
        optimizer->step();

        // 打印损失
        if ((epoch + 1) % 20 == 0) {
            std::cout << "Epoch " << epoch + 1 << ", Loss: " << loss->data[0] << std::endl;
        }
    }

    return 0;
}