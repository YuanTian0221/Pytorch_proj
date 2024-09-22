#ifndef MODULE_H
#define MODULE_H

#include <vector>
#include <memory>
#include <type_traits>
#include "tensor.h"

// 基础模块类
template <typename T>
class Module {
public:
    virtual ~Module();

    // 获取所有参数的列表
    std::vector<std::shared_ptr<Tensor<T>>> parameters();

    // 将所有参数的梯度清零
    void zero_grad();

    // 用于前向传播调用
    std::shared_ptr<Tensor<T>> operator()(const std::vector<std::shared_ptr<Tensor<T>>>& inputs);

    // 纯虚函数：子类必须实现
    virtual std::shared_ptr<Tensor<T>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) = 0;
    /*
    // 模板化的forward函数，处理不同数量的参数
    template <typename... Args>
    std::shared_ptr<Tensor<T>> forward(const Args&... args) {
        std::vector<std::shared_ptr<Tensor<T>>> inputs = {args...};
        return forward(inputs);
    }
    */
protected:
    std::vector<std::shared_ptr<void>> members_;  // 用于存储所有成员（包括参数和子模块）

    // 添加一个成员到模块中
    template <typename U>
    void register_member(const std::shared_ptr<U>& member);
};

#include "module_impl.h"  // 引入实现文件

#endif // MODULE_H
