#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <numeric>
#include <algorithm>
#include <initializer_list>
#include <cassert>
#include <set>

template <typename T>
class Tensor : public std::enable_shared_from_this<Tensor<T>> {
public:
    // 数据成员
    std::vector<T> data;  // 存储张量数据
    std::vector<size_t> shape;  // 张量的形状
    std::vector<size_t> strides;  // 用于计算一维索引的步幅
    std::vector<T> grad;  // 梯度，与 data 大小相同
    bool requires_grad;  // 是否需要计算梯度

    // 自动微分相关
    std::function<void()> _backward;  // 反向传播函数
    std::vector<std::shared_ptr<Tensor<T>>> _prev;  // 前驱节点
    std::string _op;  // 生成该张量的操作符

    // 构造函数
    Tensor(const std::vector<T>& data,
           const std::vector<size_t>& shape,
           bool requires_grad = false,
           const std::vector<std::shared_ptr<Tensor<T>>>& children = {},
           const std::string& op = "");
    
    // 支持从标量构造
    Tensor(T value, bool requires_grad = false);

    // 元素访问
    T& operator[](const std::vector<size_t>& indices);
    const T& operator[](const std::vector<size_t>& indices) const;

    // 基本算术运算
    std::shared_ptr<Tensor<T>> operator+(const std::shared_ptr<Tensor<T>>& other);
    std::shared_ptr<Tensor<T>> operator-(const std::shared_ptr<Tensor<T>>& other);
    std::shared_ptr<Tensor<T>> operator*(const std::shared_ptr<Tensor<T>>& other);
    std::shared_ptr<Tensor<T>> operator/(const std::shared_ptr<Tensor<T>>& other);

    // 反向传播
    void backward();

    // 辅助函数
    void zero_grad();  // 清零梯度
    void reshape(const std::vector<size_t>& new_shape);  // 重塑张量形状
    void print() const;  // 打印张量信息
};

// 构造函数的实现
template <typename T>
Tensor<T>::Tensor(const std::vector<T>& data,
                  const std::vector<size_t>& shape,
                  bool requires_grad,
                  const std::vector<std::shared_ptr<Tensor<T>>>& children,
                  const std::string& op)
    : data(data), shape(shape), requires_grad(requires_grad), _prev(children), _op(op) {
    // 计算 strides
    strides.resize(shape.size());
    size_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    // 初始化梯度
    if (requires_grad) {
        grad = std::vector<T>(data.size(), static_cast<T>(0));
    }
}

template <typename T>
Tensor<T>::Tensor(T value, bool requires_grad)
    : data({value}), shape({1}), requires_grad(requires_grad) {
    strides = {1};
    if (requires_grad) {
        grad = std::vector<T>(1, static_cast<T>(0));
    }
}

//实现元素的访问
template <typename T>
T& Tensor<T>::operator[](const std::vector<size_t>& indices) {
    assert(indices.size() == shape.size());
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        assert(indices[i] < shape[i]);
        offset += strides[i] * indices[i];
    }
    return data[offset];
}

template <typename T>
const T& Tensor<T>::operator[](const std::vector<size_t>& indices) const {
    assert(indices.size() == shape.size());
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        assert(indices[i] < shape[i]);
        offset += strides[i] * indices[i];
    }
    return data[offset];
}

//广播机制辅助函数
// 用于计算两个张量广播后的形状
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
//加法运算
/*
这里为了简化，假设输入张量的形状与输出张量一致。
如果要严格支持广播机制下的梯度，需要进行梯度聚合和缩减操作。
*/
template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::operator+(const std::shared_ptr<Tensor<T>>& other) {
    // 计算广播后的形状
    std::vector<size_t> result_shape = broadcast_shapes(this->shape, other->shape);
    // 计算结果数据大小
    size_t total_size = std::accumulate(result_shape.begin(), result_shape.end(), 1, std::multiplies<size_t>());
    std::vector<T> result_data(total_size);
    
    // 遍历每个位置，计算对应的加法结果
    for (size_t i = 0; i < total_size; ++i) {
        std::vector<size_t> idx(result_shape.size());
        size_t temp = i;
        for (int j = result_shape.size() -1 ; j >=0 ; --j) {
            idx[j] = temp % result_shape[j];
            temp /= result_shape[j];
        }
        T a = this->data.size() == 1 ? this->data[0] : (*this)[idx];
        T b = other->data.size() == 1 ? other->data[0] : (*other)[idx];
        result_data[i] = a + b;
    }
    
    // 创建新张量
    auto out = std::make_shared<Tensor<T>>(result_data, result_shape, this->requires_grad || other->requires_grad,
                                           std::vector<std::shared_ptr<Tensor<T>>>{this->shared_from_this(), other}, "+");
    
    // 定义反向传播函数
    out->_backward = [this, other, out]() {
        if (this->requires_grad) {
            for (size_t i = 0; i < this->data.size(); ++i) {
                this->grad[i] += out->grad[i];
                std::cout << "this data: " << this->data[i] << ", other data: " << other->data[i] << ", output grad: " << out->grad[i] << std::endl;
                std::cout << "Updating this grad: " << this->grad[i] << std::endl;
            }
        }
        if (other->requires_grad) {
            for (size_t i = 0; i < other->data.size(); ++i) {
                other->grad[i] += out->grad[i];
                std::cout << "this data: " << this->data[i] << ", other data: " << other->data[i] << ", output grad: " << out->grad[i] << std::endl;
                std::cout << "Updating this grad: " << this->grad[i] << std::endl;
            }
        }
    };
    
    return out;
}

//其他运算（减法、乘法、除法）
/*
为了完整支持广播机制下的梯度计算，
需要在反向传播中对梯度进行适当的 缩减（reduce） 操作，
这里为简化未实现。
*/
template <typename T>
std::shared_ptr<Tensor<T>> Tensor<T>::operator*(const std::shared_ptr<Tensor<T>>& other) {
    // 计算广播后的形状
    std::vector<size_t> result_shape = broadcast_shapes(this->shape, other->shape);
    // 计算结果数据大小
    size_t total_size = std::accumulate(result_shape.begin(), result_shape.end(), 1, std::multiplies<size_t>());
    std::vector<T> result_data(total_size);
    
    // 前向计算
    for (size_t i = 0; i < total_size; ++i) {
        std::vector<size_t> idx(result_shape.size());
        size_t temp = i;
        for (int j = result_shape.size() -1 ; j >=0 ; --j) {
            idx[j] = temp % result_shape[j];
            temp /= result_shape[j];
        }
        T a = this->data.size() == 1 ? this->data[0] : (*this)[idx];
        T b = other->data.size() == 1 ? other->data[0] : (*other)[idx];
        result_data[i] = a * b;
    }
    
    // 创建新张量
    auto out = std::make_shared<Tensor<T>>(result_data, result_shape, this->requires_grad || other->requires_grad,
                                           std::vector<std::shared_ptr<Tensor<T>>>{this->shared_from_this(), other}, "*");
    
    // 定义反向传播函数
    out->_backward = [this, other, out]() {
        if (this->requires_grad) {
            for (size_t i = 0; i < this->data.size(); ++i) {
                this->grad[i] += other->data[i] * out->grad[i];
                std::cout << "this data: " << this->data[i] << ", other data: " << other->data[i] << ", output grad: " << out->grad[i] << std::endl;
                std::cout << "Updating this grad: " << this->grad[i] << std::endl;
            }
        }
        if (other->requires_grad) {
            for (size_t i = 0; i < other->data.size(); ++i) {
                other->grad[i] += this->data[i] * out->grad[i];
                std::cout << "this data: " << this->data[i] << ", other data: " << other->data[i] << ", output grad: " << out->grad[i] << std::endl;
                std::cout << "Updating this grad: " << this->grad[i] << std::endl;
            }
        }
    };
    
    return out;
}

// 反向传播
template <typename T>
void Tensor<T>::backward() {
    // 检查输出是否为标量
    //assert(this->data.size() == 1 && "Can only backpropagate from a scalar value.");
    /*
    if (this->grad.empty()) {
        this->grad = std::vector<T>(this->data.size(), static_cast<T>(1));  // 初始化梯度为 1
        // 打印初始化后的梯度
        std::cout << "Initialized gradient: ";
        for (const auto& g : this->grad) {
            std::cout << g << " ";
        }
        std::cout << std::endl;
    }
    */
    this->grad = std::vector<T>(this->data.size(), static_cast<T>(1));
    std::cout << "Initialized gradient: ";
    for (const auto& g : this->grad) {
        std::cout << g << " ";
    }
    std::cout << std::endl;
    // 拓扑排序
    std::vector<std::shared_ptr<Tensor<T>>> topo;
    std::set<Tensor<T>*> visited;
    
    std::function<void(std::shared_ptr<Tensor<T>>)> build_topo = [&](std::shared_ptr<Tensor<T>> v) {
        if (visited.find(v.get()) == visited.end()) {
            visited.insert(v.get());
            for (auto child : v->_prev) {
                build_topo(child);
            }
            topo.push_back(v);
        }
    };
    
    build_topo(this->shared_from_this());
    
    // 初始化输出的梯度
    this->grad = std::vector<T>(1, static_cast<T>(1));
    
    // 反向传播
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->_backward) {
            (*it)->_backward();
        }
    }

    // 输出拓扑排序
    /*
    std::cout << "Topological order of tensors:" << std::endl;
    for (const auto& tensor : topo) {
        std::cout << "Tensor: " << tensor->data << ", op: " << tensor->_op << std::endl;
    }
    */

}
//辅助函数
//将张量的梯度清零，方便进行新的梯度计算。
template <typename T>
void Tensor<T>::zero_grad() {
    if (requires_grad) {
        std::fill(grad.begin(), grad.end(), static_cast<T>(0));
    }
}
//改变张量的形状，但不改变数据内容。
template <typename T>
void Tensor<T>::reshape(const std::vector<size_t>& new_shape) {
    size_t total_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>());
    assert(total_size == data.size() && "New shape must have the same number of elements as the original shape.");
    shape = new_shape;
    strides.resize(shape.size());
    size_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
}
//打印张量的形状和数据，方便调试和查看。
template <typename T>
void Tensor<T>::print() const {
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i != shape.size() -1 ) std::cout << ", ";
    }
    std::cout << "], data=[";
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << data[i];
        if (i != data.size() -1 ) std::cout << ", ";
    }
    std::cout << "])" << std::endl;
}
// 加法运算符重载
template <typename T>
std::shared_ptr<Tensor<T>> operator+(const std::shared_ptr<Tensor<T>>& lhs, const std::shared_ptr<Tensor<T>>& rhs) {
    return lhs->operator+(rhs);  // 调用 Tensor 类内部的 operator+
}
// 乘法运算符重载
template <typename T>
std::shared_ptr<Tensor<T>> operator*(const std::shared_ptr<Tensor<T>>& lhs, const std::shared_ptr<Tensor<T>>& rhs) {
    return lhs->operator*(rhs);  // 调用 Tensor 类内部的 operator*
}
int main() {
    // 创建张量 a 和 b
    auto a = std::make_shared<Tensor<float>>(std::vector<float>{1, 2, 3, 4}, std::vector<size_t>{2, 2}, true);
    auto b = std::make_shared<Tensor<float>>(std::vector<float>{5, 6, 7, 8}, std::vector<size_t>{2, 2}, true);
    
    // 进行计算
    auto c = a + b;  // 加法
    //auto d = a * b;  // 乘法
    //auto e = c * d;  // 组合操作
    
    // 前向结果
    c->print();
    
    // 反向传播
    c->backward();
    
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
