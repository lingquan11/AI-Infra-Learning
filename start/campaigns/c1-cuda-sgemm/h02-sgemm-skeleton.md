# C1-H02 SGEMM 工程骨架

- **栏**：动手（Muse，按 `start/prompts/hands-on.md`） | **计划用时**：1h | **先读**：无 | **被依赖**：C1-P03 起的 SGEMM 编程任务
- **一句话目标**：搭好构建和运行的骨架，之后每个 kernel 版本只需新增一个文件，我和 GPT 只专注写 kernel

## 材料
- CMake CUDA 支持：https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
- cuBLAS 文档：https://docs.nvidia.com/cuda/cublas/

## 步骤
1. 建立 `results/sgemm/`：`CMakeLists.txt`（`CMAKE_CUDA_ARCHITECTURES 120`、Release、链接 cuBLAS）、`src/kernels/`（每个版本一个文件）、`src/main.cu`（按命令行参数选择版本）、`run.sh`（构建并运行指定版本和规模）。
2. 骨架程序调用一次 `cublasSgemm` 并正常退出；kernel 位置先放一个空实现，留给 C1-P03 起由我填写。
3. 把构建加入 CI（只编译，不运行 GPU 程序）。

## 机器验收（开工前写死，AI 不得修改）
- [ ] `cmake -B build && cmake --build build` 成功
- [ ] `./run.sh cublas 1024` 正常退出并打印耗时
- [ ] 新增一个空 kernel 文件只需改 `src/kernels/` 和一处注册表，验证过一次

## 卡住时
- 链接 cuBLAS 失败：检查 CUDA Toolkit 的 `lib64` 路径，回到 C1-H01 的环境记录核对版本。
