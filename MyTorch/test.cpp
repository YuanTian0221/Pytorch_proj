#include <iostream>
#include <vector>
#include <memory>

// 假设 Function 类
template <typename T>
class Function {
public:
    std::shared_ptr<T> lanuch(const std::vector<std::shared_ptr<T>>& xs) {
        std::cout << "lanuch called" << std::endl;
        return xs[0];  // 简单返回第一个元素
    }
};

// Add 类，继承 Function
template <typename T>
class Add : public Function<T> {
public:
    // Add 特有的其他实现
};

int main() {
    Add<int> add_func;
    std::vector<std::shared_ptr<int>> inputs = { std::make_shared<int>(42) };
    
    auto result = add_func.lanuch(inputs);  // 调用 Function 的 lanuch
    std::cout << "Result: " << *result << std::endl;  // 输出结果
    
    return 0;
}