# C1-H01 环境：WSL2 + CUDA Toolkit + ncu

- **栏**：动手（Muse，按 `start/prompts/hands-on.md`） | **计划用时**：1.5h | **先读**：无 | **被依赖**：C1-P01 起的所有 CUDA 编程任务
- **可以提前做**：不必等到 12-07

## 官方文档
- NVIDIA CUDA on WSL User Guide：https://docs.nvidia.com/cuda/wsl-user-guide/index.html （"Getting Started"和"Known Limitations"）
- CUDA Toolkit 下载页，选 Linux → x86_64 → WSL-Ubuntu：https://developer.nvidia.com/cuda-downloads
- 性能计数器权限（ERR_NVGPUCTRPERM）：https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters
- Nsight Compute 系统要求（WSL 支持）：https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/system-requirements.html

## 步骤
1. Windows 端（我自己在图形界面操作）：更新到最新 NVIDIA 驱动；NVIDIA 控制面板 → 开发者 → 管理 GPU 性能计数器 → 允许所有用户访问。
2. 安装 WSL2 和 Ubuntu 24.04；在 WSL 内运行 `nvidia-smi` 能看到显卡。
3. 在 WSL 内安装 CUDA Toolkit ≥ 12.8，**使用 WSL-Ubuntu 安装包，不装 Linux 显卡驱动**；配置好 `PATH`。
4. 安装 `build-essential`、`cmake`、`git`；让 Codex 能打开 WSL 中的仓库目录。
5. 用一个最小的 kernel（Muse 可以直接写，或用 CUDA samples）验证 ncu 能采到指标。

## 机器验收（开工前写死，AI 不得修改）
- [ ] WSL 内：`nvidia-smi`、`nvcc -V`（≥ 12.8）、`ncu --version`、`cmake --version`、`g++ --version` 的输出写进 `results/hw/environment.md`
- [ ] `ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed <最小程序>` 输出了数值（不是 ERR_NVGPUCTRPERM）
- [ ] GPT（Codex）和 Muse 都能在 WSL 中的仓库目录里运行命令

## 卡住时
- ncu 报 ERR_NVGPUCTRPERM：确认第 1 步已设置，重启 Windows 和 WSL（`wsl --shutdown`）。
- 2 小时后 ncu 在 WSL 中仍采不到指标：改用 Windows 原生的 CUDA Toolkit + Nsight Compute，在 `results/hw/environment.md` 中写明，C1 的 profiling 都在 Windows 端进行。
