# C1-T02 第一个 CUDA 程序：device-info + Vector Add + 第一次 ncu

- **线**：必做 | **计划用时**：2.5h | **前置**：C1-T01 | **依赖环境**：C1-S01（WSL + CUDA Toolkit + ncu 就绪）
- **一句话目标**：亲手写出第一个 CUDA 程序，读出自己显卡的参数，并用 ncu 看到它的硬件指标

## 材料（一手来源）
- CUDA Programming Guide 的 Programming Model 部分：https://docs.nvidia.com/cuda/cuda-programming-guide/
- Nsight Compute 命令行文档：https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
- 导读：R-C1-04（做前）、R-C1-05（做后）

## 步骤
1. 【me】写 `results/env-check/device-info.cu`：用 `cudaGetDeviceProperties` 或 `cudaDeviceGetAttribute` 打印 SM 数、每个 block 的共享内存上限、每个 SM 的寄存器数、L2 大小、warp 大小、显存位宽和频率。有 deprecated 警告时，换用 attribute 接口，或用 `nvidia-smi -q` 交叉核对。
2. 【me】写 `results/env-check/vector-add.cu`：带错误检查宏（检查每次 CUDA API 调用的返回值，以及 kernel 启动后的 `cudaGetLastError`）。用 `-arch=sm_120` 编译，测试 n = 2²⁴ 和 n = 1000003（不是 block 大小的倍数），与 CPU 结果逐元素比较。
3. 【me】用 ncu 采集 vector add kernel：
   ```bash
   ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed ./vector-add
   ```
4. 【me】写 `results/hw/rtx5060ti.md`：贴第 1 步的输出；**手算**理论 FP32 峰值（SM 数 × 每 SM 的 FP32 单元数 × 2 × 频率）和理论显存带宽（写出公式），再与 NVIDIA 官方规格页对照。如果差了 2 倍或 4 倍，查明原因。

## 所有权
- `[me]`：`device-info.cu`、`vector-add.cu`、`rtx5060ti.md` 中的计算
- `[ai]`：编译命令和报错的解释（不代写 kernel）

## 机器验收（开工前写死，AI 不得修改）
- [ ] `results/hw/rtx5060ti.md` 贴有 `device-info` 的原始输出
- [ ] `vector-add` 在两个规模下都输出 PASS，命令和输出贴进 log 或 notes
- [ ] ncu 输出中有 vector add kernel 的两个指标数值（截图或文本入库）
- [ ] 理论峰值算力和带宽有公式、有数字，并与官方规格对照过

## 闭卷要点（考官据此出题）
- Host 内存和 Device 内存，以及 `cudaMemcpy` 的方向；kernel 启动是同步还是异步？错误在什么时候暴露？
- vector add 每个元素有几次 FLOP、几个字节的访存？算术强度是多少？是算力受限还是带宽受限？
- 根据你读到的 ncu 数字：DRAM 吞吐占峰值多少？离 100% 还差多少？可能的原因是什么？

## 卡住时
- 环境问题（找不到 nvcc、ncu 权限报错）：不要在这里排查，回到 C1-S01 和 Muse 一起解决。
- 超过 4h：先交付第 1、2 步，ncu 和理论峰值计算挪到 C1-T03 之前补完。
