# DetectorInference

- [DetectorInference](#detectorinference)
  - [以c++的方式运行](#以c的方式运行)

## 以c++的方式运行
```
conan profile path default
cd DetectorInference
conan install . --output-folder=build --build=missing
cmake --preset conan-default
cmake -B ./build -DCMAKE_BUILD_TYPE=Release
cmake --build ./build/ --config Release
```