# CMake 最低版本要求
cmake_minimum_required(VERSION 3.10)

# 项目名称
project(TensorProject)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)

# 添加可执行文件
add_executable(TensorApp main.cpp tensor.cpp)

# 包含头文件路径
target_include_directories(TensorApp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
