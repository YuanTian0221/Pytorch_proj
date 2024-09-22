#ifndef BASE_OP_H
#define BASE_OP_H
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>
#include "tensor.h"
#include "function.h"


// Add 类，继承自 Function，支持多个输入
template <typename T>
class Add : public Function<T> {
public:
    std::vector<size_t> out_shape;
    // 前向传播：计算多个 Tensor 的加法
    std::shared_ptr<Tensor<T>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override {
        this->inputs = inputs;  // 保存输入
        std::vector<T> result(inputs[0]->data.size(), static_cast<T>(0)); // 初始化结果
        
        // 逐元素累加
        for (size_t i = 0; i < inputs.size(); ++i) {
            for (size_t j = 0; j < result.size(); ++j) {
                result[j] += inputs[i]->data[j];
            }
        }
        // 计算输出tensor的shape
        out_shape = inputs[0]->shape;

        // 创建输出 Tensor
        auto output = std::make_shared<Tensor<T>>(result, out_shape, true, false);  // 输出 Tensor 需要计算梯度
        //output->set_creator(this->shared_from_this());  // 设置输出的 creator
        //std::cout << "Creator generation: " << output->creator->generation << std::endl;
        //std::cout << "Result generation: " << output->generation << std::endl;
        
        return output;
    }

    // 反向传播：加法的反向传播，梯度不变，所有输入的梯度都是传入的梯度
    std::vector<std::shared_ptr<Tensor<T>>> backward(const std::shared_ptr<Tensor<T>>& grad_output) override {
        std::vector<std::shared_ptr<Tensor<T>>> grad_inputs;
        
        // 对每个输入，返回相同的 grad_output
        for (size_t i = 0; i < this->inputs.size(); ++i) {
            auto grad_input = std::make_shared<Tensor<T>>(grad_output->data, grad_output->shape, false, grad_output->device);
            grad_inputs.push_back(grad_input);
        }
        
        return grad_inputs;
    }
};
// 减法
template <typename T>
class Subtraction : public Function<T> {
public:
    std::vector<size_t> out_shape;

    // 前向传播：计算两个 Tensor 的减法
    std::shared_ptr<Tensor<T>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override {
        assert(inputs.size() == 2 && "Subtraction requires exactly two input tensors");
        auto a = inputs[0];
        auto b = inputs[1];

        // 确保两个张量形状相同
        assert(a->shape == b->shape && "Tensor shapes must match for element-wise subtraction");

        this->inputs = inputs;  // 保存输入
        std::vector<T> result(a->data.size(), static_cast<T>(0)); // 初始化结果

        // 逐元素相减
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] = a->data[i] - b->data[i];
        }

        // 设置输出张量的形状
        out_shape = a->shape;

        // 创建输出 Tensor
        auto output = std::make_shared<Tensor<T>>(result, out_shape, true, false);  // 输出 Tensor 需要计算梯度
        return output;
    }

    // 反向传播：减法的反向传播，根据链式法则，返回相应的梯度
    std::vector<std::shared_ptr<Tensor<T>>> backward(const std::shared_ptr<Tensor<T>>& grad_output) override {
        std::vector<std::shared_ptr<Tensor<T>>> grad_inputs;

        // 对每个输入，返回相应的梯度
        for (size_t i = 0; i < this->inputs.size(); ++i) {
            std::vector<T> grad_data(this->inputs[i]->data.size());

            for (size_t j = 0; j < grad_data.size(); ++j) {
                if (i == 0) {
                    grad_data[j] = grad_output->data[j];  // 对第一个输入
                } else {
                    grad_data[j] = -grad_output->data[j];  // 对第二个输入
                }
            }

            auto grad_input = std::make_shared<Tensor<T>>(grad_data, this->inputs[i]->shape, false, this->inputs[i]->device);
            grad_inputs.push_back(grad_input);
        }

        return grad_inputs;
    }
};

// 乘法
template <typename T>
class Multiplication : public Function<T> {
public:
    std::vector<size_t> out_shape;

    // 前向传播：计算两个 Tensor 的乘法
    std::shared_ptr<Tensor<T>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override {
        assert(inputs.size() == 2 && "Multiplication requires exactly two input tensors");
        auto a = inputs[0];
        auto b = inputs[1];

        // 确保两个张量形状相同
        assert(a->shape == b->shape && "Tensor shapes must match for element-wise multiplication");

        this->inputs = inputs;  // 保存输入
        std::vector<T> result(a->data.size(), static_cast<T>(1)); // 初始化结果

        // 逐元素相乘
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] = a->data[i] * b->data[i];
        }

        // 设置输出张量的形状
        out_shape = a->shape;

        // 创建输出 Tensor
        auto output = std::make_shared<Tensor<T>>(result, out_shape, true, false);  // 输出 Tensor 需要计算梯度
        return output;
    }

    // 反向传播：乘法的反向传播，根据链式法则，返回相应的梯度
    std::vector<std::shared_ptr<Tensor<T>>> backward(const std::shared_ptr<Tensor<T>>& grad_output) override {
        std::vector<std::shared_ptr<Tensor<T>>> grad_inputs;

        // 对每个输入，返回相应的梯度
        for (size_t i = 0; i < this->inputs.size(); ++i) {
            std::vector<T> grad_data(this->inputs[i]->data.size());

            for (size_t j = 0; j < grad_data.size(); ++j) {
                if (i == 0) {
                    grad_data[j] = grad_output->data[j] * this->inputs[1]->data[j];  // 乘以第二个输入
                } else {
                    grad_data[j] = grad_output->data[j] * this->inputs[0]->data[j];  // 乘以第一个输入
                }
            }

            auto grad_input = std::make_shared<Tensor<T>>(grad_data, this->inputs[i]->shape, false, this->inputs[i]->device);
            grad_inputs.push_back(grad_input);
        }

        return grad_inputs;
    }
};
// 除法
template <typename T>
class Division : public Function<T> {
public:
    std::vector<size_t> out_shape;

    // 前向传播：计算两个 Tensor 的除法
    std::shared_ptr<Tensor<T>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override {
        assert(inputs.size() == 2 && "Division requires exactly two input tensors");
        auto a = inputs[0];
        auto b = inputs[1];

        // 确保两个张量形状相同
        assert(a->shape == b->shape && "Tensor shapes must match for element-wise division");

        this->inputs = inputs;  // 保存输入
        std::vector<T> result(a->data.size(), static_cast<T>(1)); // 初始化结果

        // 逐元素相除
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] = a->data[i] / b->data[i];
        }

        // 设置输出张量的形状
        out_shape = a->shape;

        // 创建输出 Tensor
        auto output = std::make_shared<Tensor<T>>(result, out_shape, true, false);  // 输出 Tensor 需要计算梯度
        return output;
    }

    // 反向传播：除法的反向传播，根据链式法则，返回相应的梯度
    std::vector<std::shared_ptr<Tensor<T>>> backward(const std::shared_ptr<Tensor<T>>& grad_output) override {
        std::vector<std::shared_ptr<Tensor<T>>> grad_inputs;

        // 对每个输入，返回相应的梯度
        for (size_t i = 0; i < this->inputs.size(); ++i) {
            std::vector<T> grad_data(this->inputs[i]->data.size());

            for (size_t j = 0; j < grad_data.size(); ++j) {
                if (i == 0) {
                    grad_data[j] = grad_output->data[j] / this->inputs[1]->data[j];  // 对第一个输入
                } else {
                    grad_data[j] = -grad_output->data[j] * this->inputs[0]->data[j] / (this->inputs[1]->data[j] * this->inputs[1]->data[j]);  // 对第二个输入
                }
            }

            auto grad_input = std::make_shared<Tensor<T>>(grad_data, this->inputs[i]->shape, false, this->inputs[i]->device);
            grad_inputs.push_back(grad_input);
        }

        return grad_inputs;
    }
};


// 矩阵乘法
template <typename T>
class MatMul : public Function<T> {
public:
    std::vector<size_t> out_shape;
    
    // 前向传播：计算矩阵乘法
    std::shared_ptr<Tensor<T>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override {
        assert(inputs.size() == 2 && "MatMul requires exactly two input tensors");
        this->inputs = inputs;  // 保存输入
        
        const auto& a = inputs[0];
        const auto& b = inputs[1];

        // 确保矩阵形状匹配
        assert(a->shape.size() == 2 && b->shape.size() == 2 && "MatMul requires 2D tensors");
        assert(a->shape[1] == b->shape[0] && "Matrix dimensions must match for multiplication");

        // 计算输出形状
        out_shape = {a->shape[0], b->shape[1]};
        std::vector<T> result(out_shape[0] * out_shape[1], static_cast<T>(0));  // 初始化结果

        // 进行矩阵乘法
        for (size_t i = 0; i < a->shape[0]; ++i) {
            for (size_t j = 0; j < b->shape[1]; ++j) {
                for (size_t k = 0; k < a->shape[1]; ++k) {
                    result[i * out_shape[1] + j] += a->data[i * a->shape[1] + k] * b->data[k * b->shape[1] + j];
                }
            }
        }

        // 创建输出 Tensor
        auto output = std::make_shared<Tensor<T>>(result, out_shape, true, false);  // 输出 Tensor 需要计算梯度
        
        return output;
    }

    // 反向传播：计算矩阵乘法的梯度
    std::vector<std::shared_ptr<Tensor<T>>> backward(const std::shared_ptr<Tensor<T>>& grad_output) override {
        std::vector<std::shared_ptr<Tensor<T>>> grad_inputs;

        const auto& a = this->inputs[0];
        const auto& b = this->inputs[1];

        // 计算输入 a 的梯度
        std::vector<T> grad_a_data(a->shape[0] * a->shape[1], static_cast<T>(0));
        for (size_t i = 0; i < a->shape[0]; ++i) {
            for (size_t j = 0; j < a->shape[1]; ++j) {
                for (size_t k = 0; k < b->shape[1]; ++k) {
                    grad_a_data[i * a->shape[1] + j] += grad_output->data[i * b->shape[1] + k] * b->data[j * b->shape[1] + k];
                }
            }
        }
        auto grad_a = std::make_shared<Tensor<T>>(grad_a_data, a->shape, false, a->device);
        grad_inputs.push_back(grad_a);

        // 计算输入 b 的梯度
        std::vector<T> grad_b_data(b->shape[0] * b->shape[1], static_cast<T>(0));
        for (size_t i = 0; i < b->shape[0]; ++i) {
            for (size_t j = 0; j < b->shape[1]; ++j) {
                for (size_t k = 0; k < a->shape[0]; ++k) {
                    grad_b_data[i * b->shape[1] + j] += a->data[k * a->shape[1] + i] * grad_output->data[k * b->shape[1] + j];
                }
            }
        }
        auto grad_b = std::make_shared<Tensor<T>>(grad_b_data, b->shape, false, b->device);
        grad_inputs.push_back(grad_b);

        return grad_inputs;
    }
};

// 广播
template <typename T>
class Broadcast : public Function<T> {
public:
    // 前向传播：广播输入张量到目标形状
    std::shared_ptr<Tensor<T>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override {
        assert(inputs.size() == 2 && "Broadcast requires an input tensor and a target shape tensor.");

        auto input = inputs[0];
        auto target_shape_tensor = inputs[1];
        assert(target_shape_tensor->shape.size() == 1 && "Target shape should be a 1D tensor.");

        // 获取目标形状
        std::vector<size_t> target_shape(target_shape_tensor->data.begin(), target_shape_tensor->data.end());

        // 广播输入张量
        return input->broadcast_to(target_shape);
    }

    // 反向传播：广播的反向传播
    std::vector<std::shared_ptr<Tensor<T>>> backward(const std::shared_ptr<Tensor<T>>& grad_output) override {
        // 反向传播时，保持广播前的梯度维度
        auto input = this->inputs[0];
        std::vector<size_t> input_shape = input->shape;

        // 初始化梯度
        auto grad_input = grad_output->sum_to(input_shape);
        return {grad_input};
    }
};


// Branch 类，继承自 Function，支持张量的分支操作
/*
前向传播 (forward):

接收一个输入张量，并复制它 num_outputs 次，每个复制的输出张量都共享相同的数据。
将 creator 设置为当前 Branch 操作，以便在反向传播时追踪到。
反向传播 (backward):

在反向传播过程中，将所有输出张量的梯度相加，并传回给输入张量。
通过 outputs 中保存的弱引用来获取输出张量的梯度，并进行累加。
输出数量 (num_outputs):

通过 set_num_outputs 方法来设置输出张量的数量。这允许用户指定分支操作需要创建多少个复制的输出张量。
*/
template <typename T>
class Branch : public Function<T> {
public:
    // 构造函数，接受输出张量的数量
    Branch(size_t num_outputs) : num_outputs(num_outputs) {}
    // 前向传播：将输入张量复制为多个输出张量
    std::vector<std::shared_ptr<Tensor<T>>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override {
        assert(inputs.size() == 1 && "Branch requires exactly one input tensor");
        std::shared_ptr<Tensor<T>> input = inputs[0];
        this->inputs = inputs;  // 保存输入

        std::vector<std::shared_ptr<Tensor<T>>> outputs;
        for (size_t i = 0; i < num_outputs; ++i) {
            // 创建输出张量，并复制输入张量的数据
            auto output = std::make_shared<Tensor<T>>(input->data, input->shape, input->requires_grad, input->device);
            output->set_creator(this->shared_from_this());  // 设置输出的 creator
            outputs.push_back(output);
        }

        return outputs;
    }

    // 反向传播：将所有上游传递回来的梯度相加，并传递给输入张量
    std::vector<std::shared_ptr<Tensor<T>>> backward(const std::shared_ptr<Tensor<T>>& grad_output) override {
        assert(this->inputs.size() == 1 && "Branch backward expects exactly one input tensor");

        auto input_grad = std::make_shared<Tensor<T>>(std::vector<T>(this->inputs[0]->data.size(), static_cast<T>(0)),
                                                      this->inputs[0]->shape, false, this->inputs[0]->device);

        // 将所有输出张量的梯度相加
        for (size_t i = 0; i < this->outputs.size(); ++i) {
            auto output_grad = this->outputs[i].lock();  // 获取弱引用的张量
            if (output_grad) {
                std::transform(input_grad->data.begin(), input_grad->data.end(),
                               output_grad->grad.begin(), input_grad->data.begin(), std::plus<T>());
            }
        }

        return {input_grad};
    }

    // 设置输出的数量
    void set_num_outputs(size_t n) {
        num_outputs = n;
    }

private:
    size_t num_outputs = 1;  // 默认输出数量为 1
};

// Repeat 操作
template <typename T>
class Repeat : public Function<T> {
public:
    int N;      // Repeat次数
    int axis;   // 重复的维度

    // 构造函数
    Repeat(int N, int axis) : N(N), axis(axis) {}

    // 前向传播
    std::shared_ptr<Tensor<T>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override {
        assert(inputs.size() == 1 && "Repeat operation takes exactly one input tensor.");
        auto input = inputs[0];
        auto input_shape = input->shape;

        // 如果 axis 超出 input 的 shape 范围，扩展 input 的 shape
        if (axis >= input_shape.size()) {
            input_shape.insert(input_shape.end(), axis - input_shape.size() + 1, 1);
        }

        // 计算新的张量形状
        std::vector<size_t> new_shape = input_shape;
        new_shape[axis] *= N;

        // 创建新的数据存储空间
        std::vector<T> new_data(std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>()));

        // 执行 repeat 操作
        size_t num_repeats = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>()) / input->data.size();
        for (size_t i = 0; i < num_repeats; ++i) {
            size_t offset = i * input->data.size();
            for (size_t j = 0; j < input->data.size(); ++j) {
                new_data[offset + j] = input->data[j];
            }
        }

        auto output = std::make_shared<Tensor<T>>(new_data, new_shape, input->requires_grad, input->device);
        output->set_creator(this->shared_from_this());
        return output;
    }

    // 反向传播
    std::vector<std::shared_ptr<Tensor<T>>> backward(const std::shared_ptr<Tensor<T>>& grad_output) override {
        auto input = this->inputs[0];
        std::vector<size_t> input_shape = input->shape;

        // 如果 axis 超出 input 的 shape 范围，扩展 input 的 shape
        if (axis >= input_shape.size()) {
            input_shape.insert(input_shape.end(), axis - input_shape.size() + 1, 1);
        }

        // 创建与原始输入大小相同的梯度
        std::vector<T> grad_input(input->data.size(), static_cast<T>(0));

        // 计算梯度：将梯度积累到原始形状中
        size_t num_repeats = grad_output->shape[axis] / input_shape[axis];
        for (size_t i = 0; i < grad_output->data.size(); ++i) {
            size_t input_index = (i / num_repeats) % input->data.size();
            grad_input[input_index] += grad_output->data[i];
        }

        return { std::make_shared<Tensor<T>>(grad_input, input_shape, input->requires_grad, input->device) };
    }
};


// Sum 操作，支持在指定维度上求和
template <typename T>
class Sum : public Function<T> {
public:
    std::vector<size_t> axes;  // 存储需要进行 sum 操作的维度
    bool keepdims;  // 是否保持降维后的维度

    // 构造函数
    Sum(const std::vector<size_t>& axes, bool keepdims = false) : axes(axes), keepdims(keepdims) {}

    // 前向传播
    std::shared_ptr<Tensor<T>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override {
        assert(inputs.size() == 1 && "Sum operation takes exactly one input tensor.");
        auto input = inputs[0];

        // 排序并去重需要进行 sum 的维度
        std::sort(axes.begin(), axes.end());
        axes.erase(std::unique(axes.begin(), axes.end()), axes.end());

        // 计算输出张量的形状
        std::vector<size_t> new_shape = input->shape;
        for (size_t axis : axes) {
            if (keepdims) {
                new_shape[axis] = 1;
            } else {
                new_shape.erase(new_shape.begin() + axis);
            }
        }

        // 初始化新的数据
        std::vector<T> new_data(std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>()), static_cast<T>(0));

        // 执行 sum 操作
        sum_recursive(input->data, new_data, input->shape, new_shape, 0, 0, axes);

        auto output = std::make_shared<Tensor<T>>(new_data, new_shape, input->requires_grad, input->device);
        output->set_creator(this->shared_from_this());
        return output;
    }

    // 反向传播
    std::vector<std::shared_ptr<Tensor<T>>> backward(const std::shared_ptr<Tensor<T>>& grad_output) override {
        auto input = this->inputs[0];
        std::vector<size_t> input_shape = input->shape;

        // 创建与原始输入大小相同的梯度
        std::vector<T> grad_input(input->data.size(), static_cast<T>(0));

        // 执行反向传播：将 grad_output 的数据广播回原始输入的形状
        broadcast_backward(grad_output->data, grad_input, grad_output->shape, input_shape, axes);

        return { std::make_shared<Tensor<T>>(grad_input, input_shape, input->requires_grad, input->device) };
    }

private:
    // 递归执行 sum 操作
    void sum_recursive(const std::vector<T>& input_data, std::vector<T>& output_data, 
                       const std::vector<size_t>& input_shape, const std::vector<size_t>& output_shape,
                       size_t input_index, size_t output_index, const std::vector<size_t>& axes) {
        if (axes.empty()) {
            output_data[output_index] += input_data[input_index];
        } else {
            size_t axis = axes[0];
            size_t stride = std::accumulate(input_shape.begin() + axis + 1, input_shape.end(), 1, std::multiplies<size_t>());
            size_t num_elements = input_shape[axis];

            for (size_t i = 0; i < num_elements; ++i) {
                sum_recursive(input_data, output_data, input_shape, output_shape,
                              input_index + i * stride, output_index, {axes.begin() + 1, axes.end()});
            }
        }
    }

    // 反向传播的广播操作
    void broadcast_backward(const std::vector<T>& grad_output, std::vector<T>& grad_input, 
                            const std::vector<size_t>& output_shape, const std::vector<size_t>& input_shape,
                            const std::vector<size_t>& axes) {
        std::vector<size_t> indices(input_shape.size());
        for (size_t i = 0; i < grad_input.size(); ++i) {
            size_t temp = i;
            for (size_t j = input_shape.size(); j-- > 0;) {
                indices[j] = temp % input_shape[j];
                temp /= input_shape[j];
            }

            size_t output_index = 0;
            size_t stride = 1;
            for (size_t j = output_shape.size(); j-- > 0;) {
                if (std::find(axes.begin(), axes.end(), j) == axes.end()) {
                    output_index += indices[j] * stride;
                    stride *= output_shape[j];
                }
            }

            grad_input[i] = grad_output[output_index];
        }
    }
};

// Sum_all
template <typename T>
class Sum_all : public Function<T> {
public:
    // 前向传播
    std::shared_ptr<Tensor<T>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override {
        assert(inputs.size() == 1 && "Sum_all operation requires exactly one input tensor.");
        auto input = inputs[0];

        // 计算所有元素的和
        T sum = std::accumulate(input->data.begin(), input->data.end(), static_cast<T>(0));

        // 创建结果 Tensor
        std::vector<size_t> output_shape = {};  // 标量，shape为空
        auto output = std::make_shared<Tensor<T>>(std::vector<T>{sum}, output_shape, input->requires_grad, input->device);
        output->set_creator(this->shared_from_this());

        return output;
    }

    // 反向传播
    std::vector<std::shared_ptr<Tensor<T>>> backward(const std::shared_ptr<Tensor<T>>& grad_output) override {
        auto input = this->inputs[0];

        // 将上游传来的梯度均分给输入 Tensor 的所有元素
        std::vector<T> grad_input(input->data.size(), grad_output->data[0]);

        return { std::make_shared<Tensor<T>>(grad_input, input->shape, input->requires_grad, input->device) };
    }
};


// Pow
template <typename T>
class Pow : public Function<T> {
public:
    T exponent;

    Pow(T exponent) : exponent(exponent) {}

    std::shared_ptr<Tensor<T>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override {
        assert(inputs.size() == 1 && "Pow operation takes exactly one input tensor.");
        auto input = inputs[0];
        std::vector<T> result(input->data.size());

        // 计算每个元素的幂
        for (size_t i = 0; i < input->data.size(); ++i) {
            result[i] = std::pow(input->data[i], exponent);
        }

        auto output = std::make_shared<Tensor<T>>(result, input->shape, input->requires_grad, input->device);
        output->set_creator(this->shared_from_this());
        return output;
    }

    std::vector<std::shared_ptr<Tensor<T>>> backward(const std::shared_ptr<Tensor<T>>& grad_output) override {
        auto input = this->inputs[0];
        std::vector<T> grad_input(input->data.size());

        // 计算反向传播的梯度
        for (size_t i = 0; i < input->data.size(); ++i) {
            grad_input[i] = exponent * std::pow(input->data[i], exponent - 1) * grad_output->data[i];
        }

        return { std::make_shared<Tensor<T>>(grad_input, input->shape, input->requires_grad, input->device) };
    }
};

// Log
template <typename T>
class Log : public Function<T> {
public:
    std::shared_ptr<Tensor<T>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override {
        assert(inputs.size() == 1 && "Log operation takes exactly one input tensor.");
        auto input = inputs[0];
        std::vector<T> result(input->data.size());

        // 计算每个元素的对数
        for (size_t i = 0; i < input->data.size(); ++i) {
            result[i] = std::log(input->data[i]);
        }

        auto output = std::make_shared<Tensor<T>>(result, input->shape, input->requires_grad, input->device);
        output->set_creator(this->shared_from_this());
        return output;
    }

    std::vector<std::shared_ptr<Tensor<T>>> backward(const std::shared_ptr<Tensor<T>>& grad_output) override {
        auto input = this->inputs[0];
        std::vector<T> grad_input(input->data.size());

        // 计算反向传播的梯度
        for (size_t i = 0; i < input->data.size(); ++i) {
            grad_input[i] = grad_output->data[i] / input->data[i];
        }

        return { std::make_shared<Tensor<T>>(grad_input, input->shape, input->requires_grad, input->device) };
    }
};

// Exp
template <typename T>
class Exp : public Function<T> {
public:
    std::shared_ptr<Tensor<T>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override {
        assert(inputs.size() == 1 && "Exp operation takes exactly one input tensor.");
        auto input = inputs[0];
        std::vector<T> result(input->data.size());

        // 计算每个元素的指数
        for (size_t i = 0; i < input->data.size(); ++i) {
            result[i] = std::exp(input->data[i]);
        }

        auto output = std::make_shared<Tensor<T>>(result, input->shape, input->requires_grad, input->device);
        output->set_creator(this->shared_from_this());
        return output;
    }

    std::vector<std::shared_ptr<Tensor<T>>> backward(const std::shared_ptr<Tensor<T>>& grad_output) override {
        auto input = this->inputs[0];
        std::vector<T> grad_input(input->data.size());

        // 计算反向传播的梯度
        for (size_t i = 0; i < input->data.size(); ++i) {
            grad_input[i] = grad_output->data[i] * std::exp(input->data[i]);
        }

        return { std::make_shared<Tensor<T>>(grad_input, input->shape, input->requires_grad, input->device) };
    }
};

// Neg
template <typename T>
class Neg : public Function<T> {
public:
    std::shared_ptr<Tensor<T>> forward(const std::vector<std::shared_ptr<Tensor<T>>>& inputs) override {
        assert(inputs.size() == 1 && "Neg operation takes exactly one input tensor.");
        auto input = inputs[0];
        std::vector<T> result(input->data.size());

        // 计算每个元素的负值
        for (size_t i = 0; i < input->data.size(); ++i) {
            result[i] = -input->data[i];
        }

        auto output = std::make_shared<Tensor<T>>(result, input->shape, input->requires_grad, input->device);
        output->set_creator(this->shared_from_this());
        return output;
    }

    std::vector<std::shared_ptr<Tensor<T>>> backward(const std::shared_ptr<Tensor<T>>& grad_output) override {
        auto input = this->inputs[0];
        std::vector<T> grad_input(input->data.size());

        // 计算反向传播的梯度
        for (size_t i = 0; i < input->data.size(); ++i) {
            grad_input[i] = -grad_output->data[i];
        }

        return { std::make_shared<Tensor<T>>(grad_input, input->shape, input->requires_grad, input->device) };
    }
};


#endif
