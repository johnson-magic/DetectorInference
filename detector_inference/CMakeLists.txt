cmake_minimum_required(VERSION 3.10)
project(detector_inference_library)
find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)
set(DETECTOR_INFERENCE_LIBRARY
    src/detector_inference.cpp
)
add_library(detector_inference ${DETECTOR_INFERENCE_LIBRARY})

include_directories("./include")
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${onnxruntime_INCLUDE_DIR})

set_target_properties(detector_inference PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES)  # 这个是必须的，否则会报错

set_target_properties(detector_inference PROPERTIES PUBLIC_HEADER "include/data_structure.h;include/detector_inference.h")
install(TARGETS detector_inference)