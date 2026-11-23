# C1-T01 环境打通 + Vector Add + 第一次 ncu

- **线**：必做 | **计划用时**：3h | **前置**：无
- **一句话目标**：在 WSL2 中写一个 CUDA 程序，编译、运行、用 ncu 采到硬件指标；把整个链路的风险压到第一周

## 材料（一手来源）
- NVIDIA CUDA on WSL User Guide：https://docs.nvidia.com/cuda/wsl-user-guide/index.html （读"Getting Started"和"Known Limitations"）
- CUDA Toolkit 下载页，选 Linux → x86_64 → WSL-Ubuntu：https://developer.nvidia.com/cuda-downloads
- 性能计数器权限（ERR_NVGPUCTRPERM）：https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters
- Nsight Compute 命令行文档：https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
- PyTorch 安装页：https://pytorch.org/get-started/locally/
- CUDA Programming Guide 的 Programming Model 部分：https://docs.nvidia.com/cuda/cuda-programming-guide/

## 步骤
1. Windows 端：更新到最新 NVIDIA 驱动；NVIDIA 控制面板 → 开发者 → 管理 GPU 性能计数器 → 允许所有用户访问。
2. 安装 WSL2（Ubuntu 24.04），在 WSL 内运行 `nvidia-smi` 确认能看到显卡。
3. 在 WSL 内安装 CUDA Toolkit ≥ 12.8，使用 **WSL-Ubuntu** 安装包（不要装 Linux 显卡驱动）。确认 `nvcc -V` 和 `ncu --version` 都能运行。
4. 【me】写 `results/env-check/device-info.cu`：用 `cudaGetDeviceProperties` 或 `cudaDeviceGetAttribute` 打印 SM 数、每个 block 的共享内存上限、每个 SM 的寄存器数、L2 大小、warp 大小、显存位宽和频率。字段有 deprecated 警告时，换用 attribute 接口或以 `nvidia-smi -q` 交叉核对。
5. 【me】写 `results/env-check/vector-add.cu`：带错误检查宏（检查每次 CUDA API 调用的返回值，以及 kernel 启动后的 `cudaGetLastError`）。用 `-arch=sm_120` 编译，测试 n = 2²⁴ 和 n = 1000003（不是 block 大小的倍数），与 CPU 结果逐元素比较。
6. 【me】用 ncu 采集 vector add kernel：
   ```bash
   ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed ./vector-add
   ```
7. 【me】在 Python 虚拟环境中安装 PyTorch（cu128 版本），运行：
   ```bash
   python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"
   ```
8. 【me】写 `results/hw/rtx5060ti.md`：贴第 4 步的输出；**手算**理论 FP32 峰值（SM 数 × 每 SM 的 FP32 单元数 × 2 × 频率）和理论显存带宽（写出公式），再与 NVIDIA 官方规格页对照。如果差了 2 倍或 4 倍，查明原因。

## 所有权
- `[me]`：`device-info.cu`、`vector-add.cu`、`results/hw/rtx5060ti.md` 中的计算
- `[ai]`：安装过程中的报错排查（只给建议，命令由我执行）

## 机器验收（开工前写死，AI 不得修改）
- [ ] `results/hw/rtx5060ti.md` 贴有以下命令的原始输出：`nvcc -V`（≥ 12.8）、`nvidia-smi`、PyTorch 检查（包含 `(12, 0)`）、`device-info`
- [ ] `vector-add` 在两个规模下都输出 PASS，命令和输出贴进 log 或 notes
- [ ] ncu 输出中有 vector add kernel 的两个指标数值（截图或文本入库）
- [ ] 理论峰值算力和带宽有公式、有数字，并与官方规格对照过

## 闭卷要点
- `blockIdx`、`blockDim`、`threadIdx` 怎样组合成全局下标？为什么必须加边界检查？
- Host 内存和 Device 内存，以及 `cudaMemcpy` 的方向；kernel 启动是同步还是异步？错误会在什么时候暴露？
- vector add 每个元素有几次 FLOP、几个字节的访存？它的算术强度是多少？是算力受限还是带宽受限？
- 根据你读到的 ncu 数字：DRAM 吞吐占峰值多少？离 100% 还差多少？可能的原因是什么？

## 卡住时
- `ncu` 报 ERR_NVGPUCTRPERM：确认第 1 步已经设置，重启 Windows 和 WSL（`wsl --shutdown`）。
- 超过 5h，或 ncu 在 WSL 中始终采不到指标：改用 Windows 原生的 CUDA Toolkit + Nsight Compute 完成第 5–6 步，在 log 中写明原因，C1 后续任务的 profiling 都在 Windows 端进行。不要因为这一步卡住整个战役。
