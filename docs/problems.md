1. 问题现象：error C2664: “Ort::Session: 
:Session(Ort::Session &&)”: 无法将参数 2 从“const _Elem *”转换为“const wchar_t *”
    * 解决:见参考资料,另，该问题仅在windows下出现，linux下无该现象
    * 参考资料:
      * [最终网上解决方案](https://github.com/ami-iit/bipedal-locomotion-framework/commit/91ad0cff1d2e756145ad11612a0f090c72e9f02e)
      * [相似问题官方讨论](https://github.com/microsoft/onnxruntime/issues/15889)

2. windowx下的C++构建系统
   * 解决：见参考资料
   * 参考资料：
     * [Opencv依赖](https://www.youtube.com/watch?v=CnXUTG9XYGI)
     * [Onnx依赖](https://blog.csdn.net/weixin_43953700/article/details/124304712)
     * [other](https://www.youtube.com/watch?v=oC69vlWofJQ)、[other](https://www.youtube.com/watch?v=aMXQshF7zdo)、[other](https://code.visualstudio.com/docs/cpp/config-mingw)、[other](https://code.visualstudio.com/docs/languages/cpp)
3. 问题现象：The given version [11] is not supported, only version 1 to 7 is supported in this build.
   * 解决：见参考资料
   * 参考资料：
     * [windows下Onnx库的使用](https://blog.csdn.net/weixin_43953700/article/details/124304712)