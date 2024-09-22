#ifndef TENSOR_IMPL_H
#define TENSOR_IMPL_H

#include "tensor.h"  // 包含声明文件

// 构造函数：初始化张量数据、形状、步幅等
template <typename T>
Tensor<T>::Tensor(const std::vector<T>& data, 
                  const std::vector<size_t>& shape, 
                  bool requires_grad, 
                  bool device)
    : data(data), shape(shape), requires_grad(requires_grad), device(device) {
    // 计算 strides，用于一维索引转换
    strides.resize(shape.size());
    strides.back() = 1;
    for (int i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    // 如果需要计算梯度，则初始化 grad 与 data 同大小
    if (requires_grad) {
        grad.resize(data.size(), static_cast<T>(0));
    }
}

// 标量构造函数
template <typename T>
Tensor<T>::Tensor(T value, bool requires_grad, bool device)
    : Tensor(std::vector<T>{value}, {1}, requires_grad, device) {}

// 设置创建者 (用于自动微分追踪)
template <typename T>
void Tensor<T>::set_creator(const std::shared_ptr<Function<T>>& func) {
    creator = func;

    // 计算该张量的 generation 值
    // 它应该是所有输入张量中最大的 generation + 1
    /*
    int max_generation = 0;
    for (const auto& input : func->inputs) {
        if (input) {  // 确保输入不为空
            max_generation = std::max(max_generation, input->generation);
        }
    }
    generation = max_generation + 1;  // 新的张量 generation = 前驱的最大 generation + 1
    */
}

// 张量元素访问操作符 (支持多维索引)
template <typename T>
T& Tensor<T>::operator[](const std::vector<size_t>& indices) {
    assert(indices.size() == shape.size());
    size_t index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        index += strides[i] * indices[i];
    }
    return data[index];
}

template <typename T>
const T& Tensor<T>::operator[](const std::vector<size_t>& indices) const {
    assert(indices.size() == shape.size());
    size_t index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        index += strides[i] * indices[i];
    }
    return data[index];
}


// backward
template <typename T>
void Tensor<T>::backward() {
    if (!requires_grad) {
        throw std::runtime_error("This tensor does not require gradients");
    }
    /*
    if (grad.empty()) {
        grad.resize(data.size(), static_cast<T>(1)); // 初始化梯度为 1，适用于标量
    }
    */
    // 如果当前 Tensor 是反向传播的起点
    grad.assign(data.size(), static_cast<T>(1));  // 初始化梯度为 1
    // 使用 lambda 函数进行 generation 排序
    auto compare_generation = [](const std::shared_ptr<Function<T>>& a, const std::shared_ptr<Function<T>>& b) {
        return a->generation < b->generation; // 按 generation 排序，较大的先处理
    };

    // Priority queue (heap) to sort Functions based on their generation (higher generation processed first)
    std::priority_queue<std::shared_ptr<Function<T>>, std::vector<std::shared_ptr<Function<T>>>, decltype(compare_generation)> funcs(compare_generation);
    std::set<std::shared_ptr<Function<T>>> seen;  // 防止重复处理

    // Add a function to the heap if it's not already seen
    auto add_func = [&](const std::shared_ptr<Function<T>>& f) {
        if (seen.find(f) == seen.end()) {
            funcs.push(f);  // 加入堆中
            seen.insert(f); // 标记为已处理
        }
    };

    // Start from the current tensor's creator function
    if (creator) {
        add_func(creator);
    }

    // Process the heap (topological sorting by generation)
    while (!funcs.empty()) {
        auto f = funcs.top();  // 从堆中取出 generation 最大的 Function
        funcs.pop();

        // Get the output gradient for the function
        std::shared_ptr<Tensor<T>> grad_output = std::make_shared<Tensor<T>>(grad, shape, false, device);
        auto grads = f->backward(grad_output);  // 调用反向传播，计算梯度

        // 输出当前 function 节点的 generation 值
        // std::cout << "Processing function with generation: " << f->generation << std::endl;

        // Iterate over inputs and accumulate gradients
        for (size_t i = 0; i < f->inputs.size(); ++i) {
            auto input = f->inputs[i];
            //std::cout << "Input data: " << input->data[0] << std::endl;
            if (input->requires_grad) {
                // 初始化梯度
                if (input->grad.empty()) {
                    input->grad = grads[i]->data;
                } else {
                    // 累加梯度
                    std::transform(input->grad.begin(), input->grad.end(), grads[i]->data.begin(), input->grad.begin(), std::plus<T>());
                }

                // 将 input 的 creator 添加到堆中
                if (input->creator) {
                    //std::cout << "Input data: " << input->data[0] << std::endl;
                    //std::cout << "Input Creator with generation: " << input->creator->generation << std::endl;
                    add_func(input->creator);
                }
            }
        }
    }
}


// 清零梯度
template <typename T>
void Tensor<T>::zero_grad() {
    if (requires_grad) {
        std::fill(grad.begin(), grad.end(), static_cast<T>(0));
    }
}

// 打印张量内容
template <typename T>
void Tensor<T>::print() const {
    std::cout << "Tensor(";
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << data[i] << (i < data.size() - 1 ? ", " : "");
    }
    std::cout << ")" << std::endl;
}

// 张量转置
template <typename T>
void Tensor<T>::transpose() {
    if (shape.size() != 2) {
        throw std::invalid_argument("Only 2D tensors can be transposed.");
    }

    // 交换行和列的形状
    std::swap(shape[0], shape[1]);

    std::vector<T> new_data(data.size());

    // 执行转置操作
    for (size_t i = 0; i < shape[1]; ++i) { // 注意这里用交换后的形状
        for (size_t j = 0; j < shape[0]; ++j) {
            new_data[j * shape[1] + i] = data[i * shape[0] + j];
        }
    }

    // 更新张量数据
    data = std::move(new_data);

    // 重新计算 strides
    strides[0] = shape[1];
    strides[1] = 1;
}

// 广播
/*
该函数根据目标形状创建一个新的张量，逐元素复制原始张量的数据。
需要检查原始张量的形状是否可以广播到目标形状。
*/
template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::broadcast_to(const std::vector<size_t>& target_shape) const {
    // 检查是否可以广播
    if (shape.size() > target_shape.size()) {
        throw std::invalid_argument("Cannot broadcast to a smaller number of dimensions.");
    }

    std::vector<size_t> new_shape = target_shape;
    std::vector<T> new_data(std::accumulate(target_shape.begin(), target_shape.end(), 1, std::multiplies<size_t>()));

    // 逐元素拷贝数据
    for (size_t i = 0; i < new_data.size(); ++i) {
        std::vector<size_t> indices(target_shape.size());
        size_t temp = i;
        for (size_t j = target_shape.size(); j > 0; --j) {
            indices[j - 1] = temp % target_shape[j - 1];
            temp /= target_shape[j - 1];
        }

        size_t old_index = 0;
        for (size_t j = 0; j < shape.size(); ++j) {
            if (shape[j] != 1) {
                old_index = old_index * shape[j] + indices[indices.size() - shape.size() + j];
            }
        }

        new_data[i] = data[old_index];
    }

    return std::make_shared<Tensor<T>>(new_data, new_shape, requires_grad, device);
}

// 广播的还原
/*
该函数用于将张量的形状还原到目标形状，主要用于反向传播中，将广播操作还原。
它通过求和将张量的高维度折叠到目标形状。
*/
template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::sum_to(const std::vector<size_t>& target_shape) const {
    if (shape == target_shape) {
        // 如果形状一致，直接返回一个拷贝
        return std::make_shared<Tensor<T>>(*this);
    }

    if (target_shape.size() > shape.size()) {
        throw std::invalid_argument("Target shape has more dimensions than tensor shape.");
    }

    std::vector<size_t> new_shape = target_shape;
    std::vector<T> new_data(std::accumulate(target_shape.begin(), target_shape.end(), 1, std::multiplies<size_t>()));

    for (size_t i = 0; i < data.size(); ++i) {
        std::vector<size_t> indices(shape.size());
        size_t temp = i;
        for (size_t j = shape.size(); j > 0; --j) {
            indices[j - 1] = temp % shape[j - 1];
            temp /= shape[j - 1];
        }

        size_t new_index = 0;
        for (size_t j = 0; j < target_shape.size(); ++j) {
            new_index = new_index * target_shape[j] + indices[indices.size() - target_shape.size() + j];
        }

        new_data[new_index] += data[i];
    }

    return std::make_shared<Tensor<T>>(new_data, new_shape, requires_grad, device);
}


#endif  // TENSOR_IMPL_H
