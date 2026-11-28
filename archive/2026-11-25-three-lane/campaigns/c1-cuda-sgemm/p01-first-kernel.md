# C1-P01 第一个 CUDA 程序：device-info + vector add

- **栏**：编程 | **计划用时**：2.5h | **先读**：C1-R01、R02（GPU 基础、CUDA 第 1 章） | **依赖动手**：C1-H01（WSL + CUDA + ncu 就绪）
- **一句话目标**：亲手写出第一个 CUDA 程序，读出自己显卡的参数，并知道它的理论上限

## 材料（一手来源）
- CUDA Programming Guide 的编程模型部分：https://docs.nvidia.com/cuda/cuda-programming-guide/

## 步骤
1. 【me】`results/env-check/device-info.cu`：用 `cudaGetDeviceProperties` 或 `cudaDeviceGetAttribute` 打印 SM 数、每个 block 的共享内存上限、每个 SM 的寄存器数、L2 大小、warp 大小、显存位宽和频率。
2. 【me】`results/env-check/vector-add.cu`：带错误检查宏（每次 CUDA API 调用的返回值，以及 kernel 启动后的 `cudaGetLastError`）；`-arch=sm_120` 编译；测试 n = 2²⁴ 和 n = 1000003（不是 block 大小的倍数），与 CPU 结果逐元素比较。
3. 【me】错误注入：把边界检查删掉或下标写错，确认测试报 FAIL，再改回来。
4. 【me】`results/hw/rtx5060ti.md`：贴第 1 步的输出，**手算**理论 FP32 峰值（SM 数 × 每 SM 的 FP32 单元数 × 2 × 频率）和理论显存带宽（写出公式），再与 NVIDIA 官方规格页对照；差 2 倍或 4 倍时查明原因。
5. GPT 运行编译和测试；把 vector add 交给 Muse 在 C1-H03 之前顺便跑一次 ncu（不在本任务验收内）。

## 机器验收（开工前写死，AI 不得修改）
- [ ] `device-info` 的原始输出贴进 `results/hw/rtx5060ti.md`
- [ ] `vector-add` 在两个规模下都输出 PASS；错误注入那次输出 FAIL（终端输出贴进 `results/env-check/README.md`）
- [ ] 理论峰值算力和带宽有公式、有数字，并与官方规格对照过

## 卡住时
- 找不到 `nvcc`、驱动报错：不是本任务的问题，回到 C1-H01 找 Muse。
- 超过 4h：先交付第 1–3 步，理论峰值计算挪到 C1-P02 之前补完。
