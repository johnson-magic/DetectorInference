cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(ONNXInference)

set(CMAKE_PREFIX_PATH /usr/local/share/opencv4)
# 查找 OpenCV 包
find_package(OpenCV REQUIRED)


# find_package(onnxruntime REQUIRED)
include_directories("/home/ecs-user/ultralytics/examples/YOLOv8-ONNXRuntime-CPP/onnxruntime-linux-x64-1.15.1/include/")

# 设置可执行文件名
add_executable(onnx_inference main.cpp) # main.cpp 为你的源文件名称

# 链接 OpenCV 库
target_link_libraries(onnx_inference ${OpenCV_LIBS} "/home/ecs-user/ultralytics/examples/YOLOv8-ONNXRuntime-CPP/onnxruntime-linux-x64-1.15.1/lib/libonnxruntime.so")

# 设置 C++ 标准
set_target_properties(onnx_inference PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES)
