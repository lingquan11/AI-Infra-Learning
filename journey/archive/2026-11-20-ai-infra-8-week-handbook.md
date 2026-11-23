> **归档说明（2026-11-23）**：这是本仓库路线的原始版本，已被 [start/ROADMAP.md](../../start/ROADMAP.md) 取代，保留作为出处。与 ROADMAP 冲突时以 ROADMAP 为准；改动见 ROADMAP 末尾的"变更记录"。

# AI Infra 底层能力重构与 8 周实战手册（RTX 5060 Ti）

> **核心原则**：彻底废弃“看网课—记笔记”的被动学习模式，全程采用 **「一手仓库 / 原论文 → 独立手写复现 → 硬件指标（Profiling）验收」** 的工程闭环。
>
> **节奏基准**：前期每周投入 8–10 小时（兼顾校内主课与 GPA），后期集中攻坚。所有任务均绑定可公开验证的 GitHub 交付物。

---

## 一、开工前基线：RTX 5060 Ti (`sm_120`) 环境配置清单

RTX 5060 Ti 为 NVIDIA Blackwell 架构（Compute Capability 12.0 / `sm_120`）。旧版 CUDA（12.1 / 12.6）与旧版 PyTorch 预编译包不包含 `sm_120` 内核，运行时会报 `no kernel image is available`。请按以下标准一次性配好环境：

- [ ] **宿主机驱动**：Windows 端安装最新 NVIDIA 显卡驱动。
- [ ] **Linux 子系统**：安装 WSL2（推荐 Ubuntu 24.04），后续全部开发、编译与 Profiling 均在 WSL2 内完成。
- [ ] **CUDA Toolkit**：在 WSL2 内安装 **CUDA Toolkit 12.8 及以上版本**（注意：WSL2 内**严禁**再安装 Linux 显卡驱动，仅装 Toolkit）。
- [ ] **PyTorch 版本**：在虚拟环境中安装 **PyTorch 2.8.0+cu128 及以上版本**。
- [ ] **开启硬件计数器权限**：在 Windows 端打开 `NVIDIA 控制面板` → `开发者` → `管理 GPU 性能计数器` → 勾选 **“允许所有用户访问 GPU 性能计数器”**（否则 `ncu` 无法采集硬件指标）。

### 环境自检命令（在 WSL2 终端执行）

```bash
# 1. 检查 CUDA 编译器版本（需 >= 12.8）
nvcc -V

# 2. 检查 Nsight Compute 是否可用
ncu --version

# 3. 检查 PyTorch 是否正确识别 5060 Ti (sm_120)
python3 -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda, '| Arch:', torch.cuda.get_device_capability())"
# 预期输出包含: Arch: (12, 0)
```

---

## 二、8 周实战路线总览

| 周期 | 阶段目标 | 核心锚点（仓库 / 论文 / 教材） | 硬性验收交付物 |
| :--- | :--- | :--- | :--- |
| **第 1–2 周** | C++ 矩阵乘法仓库封箱与总结 | 现有 C++ 矩阵乘法仓库 + WSL2 环境 | 发布 `v1.0` Release，补齐消融总表与 `SUMMARY.md`；5060 Ti 跑通 CUDA 基准 |
| **第 3–5 周** | CUDA 算子优化与 GPU 微架构直觉 | `srush/GPU-Puzzles`<br>`siboehm/SGEMM_CUDA`<br>*PMPP (4th Ed.)* 第 3–6 章 | 新建 `cuda-sgemm` 仓库，完成 6 阶优化，性能达 **cuBLAS 的 70%+**，附 `ncu` 剖析报告 |
| **第 6–7 周** | Triton 算子融合与 FlashAttention-2 | `gpu-mode/Triton-Puzzles`<br>`triton-lang/triton` Tutorials<br>*FlashAttention-2* (`arXiv:2307.08691`) | 手写 Fused Softmax、Fused RMSNorm 与简化版 FlashAttention-2 Forward，附延迟与显存对比曲线 |
| **第 8 周** | 轻量推理引擎源码拆解与接入 | `GeeeekExplorer/nano-vllm`<br>*PagedAttention* (SOSP 2023) | 本地跑通 `Qwen3-0.6B` 压测，手绘调度与显存流转图，完成 1 次算子替换或系统消融实验 |

---

## 三、分阶段可执行任务清单

### Phase 0（第 1–2 周）：C++ 矩阵乘法仓库封箱 + 环境打通

**目标**：把已有工作沉淀为完整的技术资产，不留“半成品尾巴”，同时打通 GPU 开发环境。

- [ ] **整理消融总表**：在现有仓库 README 中补齐各优化版本（朴素循环 → 内存分块 → SIMD → 硬件仿真等）的统一对比表，注明编译选项（如 `-O3 -march=native`）、测试矩阵规模、GFLOPS 及相对基线加速比。
- [ ] **撰写技术复盘（`SUMMARY.md`，控制在 1,000 字以内）**：聚焦回答三个问题：
  1. 性能瓶颈是如何从算力逐步转移到访存的？
  2. 每一步优化具体利用了 CPU 的什么微架构机制（Cache Line、预取、寄存器/SIMD 宽度）？
  3. CPU 优化在哪些地方碰到了物理上限？
- [ ] **归档发布**：确认一键复现命令无误，打上 `v1.0` Git Tag / Release。
- [ ] **打通 CUDA 最小闭环**：完成上文 5060 Ti 环境配置，在 WSL2 中手写并编译运行一个 CUDA Vector Add 程序。

---

### Phase 1（第 3–5 周）：CUDA SGEMM 阶梯优化（核心攻坚）

**目标**：把 CPU 访存与向量化直觉，完整迁移到 GPU 的 SIMT 执行模型与显存层级（GMEM → L2 → SMEM → Register）。

#### 锚点资源
* **热身题库**：GitHub `srush/GPU-Puzzles`（14 题）
* **主线蓝本**：Simon Boehm 博客 *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance* 及仓库 `siboehm/SGEMM_CUDA`
* **按需查阅**：*Programming Massively Parallel Processors (PMPP, 4th Ed.)* 第 3–6 章（重点看线程映射、内存层级与性能权衡）

#### 执行清单（禁止直接 Clone 跑代码，必须新建 `cuda-sgemm` 仓库独立手敲）
- [ ] **第 3 周前半周**：刷完 `srush/GPU-Puzzles` 全部 14 题，彻底搞懂 Thread/Block 索引、边界保护（Guards）与 Shared Memory 同步。
- [ ] **搭建工程框架**：新建仓库，配置 `CMakeLists.txt`（设置 `CMAKE_CUDA_ARCHITECTURES 120`），写好统一的 cuBLAS 基准测试与数值校验脚本（要求与 cuBLAS 最大误差 $< 10^{-3}$）。
- [ ] **按阶实现 6 个版本**（基准测试矩阵尺寸统一为 $4096 \times 4096$，FP32）：
  - [ ] `v1`：Naive SGEMM（建立最低性能基线）
  - [ ] `v2`：Global Memory Coalescing（全局内存合并访问）
  - [ ] `v3`：Shared Memory Caching / 2D Tiling（片上共享内存分块）
  - [ ] `v4`：1D Blocktiling（单线程计算多元素，提升算术强度）
  - [ ] `v5`：2D Blocktiling（寄存器级分块，大幅减少 SMEM 访问次数）
  - [ ] `v6`：`float4` 向量化访存与转置优化（缓解内存管线停顿）
  - [ ] *（进阶选做）* `v7`：消除 Bank Conflict 与双缓冲（Double Buffering）预取
- [ ] **硬件指标验收**：
  - [ ] 在 RTX 5060 Ti 上，`v6` 性能达到同尺寸 **cuBLAS 的 70% 以上**。
  - [ ] 使用 `ncu` 采集 `v1`、`v3`、`v6` 三个版本的剖析报告，在 README 中结合截图解释 **Compute Throughput (SOL)**、**Memory Throughput (SOL)** 与 **Warp Stall Reasons** 的变化逻辑。

---

### Phase 2（第 6–7 周）：Triton 算子融合 + FlashAttention-2 论文复现

**目标**：掌握工业界主流的 Block 级算子编写工具 Triton，完成第一篇顶会系统论文的“数学推导 → 算子落地”。

#### 锚点资源
* **热身题库**：GitHub `gpu-mode/Triton-Puzzles` + 官方 `triton-lang/triton` 前 3 个 Tutorials（Vector Add、Fused Softmax、Matmul）
* **精读论文**：Tri Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*（`arXiv:2307.08691`）

#### 执行清单
- [ ] **Triton 思维切换（3 天）**：完成 `Triton-Puzzles` 前 8–10 题与官方前 3 个教程，适应从 CUDA“单线程视角”切换到 Triton“Block 张量视角”。
- [ ] **手写访存密集型融合算子**：
  - [ ] 实现 **Fused Softmax** 与 **Fused RMSNorm**。
  - [ ] 测试不同隐藏层维度下的有效显存带宽（GB/s），与 PyTorch Eager 和 `torch.compile` 对比，计算实测带宽占 RTX 5060 Ti 理论峰值带宽的百分比。
- [ ] **精读 FlashAttention-2 论文（3 天）**：
  - [ ] 独立推演论文 **Section 2.3** 的 Online Softmax 递推公式（理解如何在不物化 $N \times N$ 矩阵的前提下保持数值稳定）。
  - [ ] 吃透 **Section 3 (Algorithm 1)** 的 Forward 伪代码及序列维并行切分策略。
- [ ] **复现 FlashAttention-2 Forward**：
  - [ ] 用 Triton 从零实现支持 Causal Mask 的简化版 FlashAttention-2 Forward。
  - [ ] **精度验收**：与 `torch.nn.functional.scaled_dot_product_attention` 对比，前向输出通过 `torch.allclose(atol=1e-2, rtol=1e-2)`。
  - [ ] **性能与显存验收**：绘制序列长度 $512 / 1024 / 2048 / 4096$ 下的**前向延迟曲线**与**峰值显存曲线**（对比标准 PyTorch 手写 Attention，验证显存占用从 $O(N^2)$ 降至 $O(N)$）。

---

### Phase 3（第 8 周）：轻量推理引擎源码拆解与系统级实战

**目标**：跳出单 Kernel 视角，完整理解现代 LLM 推理服务的显存管理（KV Cache）与请求调度机制。

#### 锚点资源
* **论文选读**：vLLM 原论文 *Efficient Memory Management for Large Language Model Serving with PagedAttention*（SOSP 2023，重点精读前 4 节）
* **核心仓库**：GitHub `GeeeekExplorer/nano-vllm`（约 1,200 行纯净 Python，完整涵盖 Prefix Caching、PagedAttention、Continuous Batching 与 CUDA Graph）

#### 执行清单
- [ ] **跑通本地基准**：下载 `Qwen/Qwen3-0.6B` 权重，在 5060 Ti 上成功运行 `nano-vllm` 的 `example.py` 与 `bench.py`。
- [ ] **源码精读与架构绘图**：逐行阅读 `LLMEngine`、`Scheduler`、`BlockManager` 和 `ModelRunner` 四个核心模块，手画一张完整数据流图（覆盖：请求入队 → Prefill/Decode 调度决策 → 逻辑块/物理块 KV Cache 映射 → 模型前向与采样返回）。
- [ ] **动手改代码与实验（二选一，有余力可全做）**：
  - [ ] **选项 A（算子接入）**：将 `nano-vllm` 模型层中的 RMSNorm 替换为你在 Phase 2 手写的 Triton Fused RMSNorm，验证输出 Token 严格一致，并对比端到端吞吐变化。
  - [ ] **选项 B（系统消融）**：控制变量跑一组压测对比（如开关 CUDA Graph `enforce_eager=True/False`、调整并发 Batch Size），绘制并分析 **TTFT（首字延迟）**、**TPOT（每 Token 延迟）** 与 **总吞吐（tokens/s）** 的权衡曲线。
- [ ] **成果归档**：整理《`nano-vllm` 源码剖析与 5060 Ti 实测报告》推送至 GitHub。

---

## 四、常用 Profiling 命令速查

在 Phase 1–3 中，不要靠肉眼猜瓶颈，养成直接抓硬件数据的习惯：

```bash
# 1. 编译针对 RTX 5060 Ti (sm_120) 的 CUDA 程序
nvcc -O3 -arch=sm_120 sgemm.cu -lcublas -o sgemm

# 2. 使用 ncu 抓取完整性能剖析报告（生成 .ncu-rep 文件，可用 Windows 端 Nsight Compute GUI 打开可视化查看）
ncu --set full -o profile_v6 ./sgemm

# 3. 快速在终端查看关键吞吐指标（无需打开 GUI）
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed ./sgemm
```

---

## 五、长期演进规划与“底子够硬”自检标准

这 8 周对应你新学习体系的 **Stage 1（单卡计算与推理底层闭环）**。完成后可按以下节奏向更深层推进：

* **Stage 1（第 1–2 个月，本手册）**：CPU 访存 → CUDA SGEMM → Triton 融合算子 → 极简推理引擎拆解。
* **Stage 2（第 3–5 个月）**：利用 5060 Ti 原生 Blackwell 硬件特性，切入 **FP8 / INT4 权重量化** 与 **Tensor Core（WMMA / CuTe / CUTLASS 入门）**；精读 `vLLM` 或 `SGLang` 某一核心模块，尝试向社区提交首个 Bug Fix 或性能优化 PR。
* **Stage 3（第 6 个月起）**：带着明确的底层问题与硬核作品集，冷静评估下一步选择——进入高校强课题组深耕体系结构/AI 系统科研，或直接对接一线 AI Lab / N1 等平台。

### 检验“是否真正具备底层实力”的 3 条硬标准

未来无论何时重新评估是否要 Gap、去 N1 或投递顶级实验室，请直接用这三条对标：

1. **独立手写能力**：给出一个新算子公式（如新型 Attention 或 Normalization 变体），不靠抄袭现成实现，能在 2 天内写出数值正确且具备较高带宽利用率的 Triton / CUDA 版本。
2. **硬件剖析直觉**：面对一段跑得慢的模型推理或训练代码，能直接用 `ncu` 或 PyTorch Profiler 抓指标，准确归因当前是算力受限（Compute-bound）、显存带宽受限（Memory-bound），还是调度/Kernel Launch 开销受限。
3. **公开证据链**：GitHub 上沉淀了连续、可一键复现的硬核工程链条（CPU 矩阵优化 → CUDA 算子阶梯优化 → Triton 论文复现 → 推理系统拆解/优化），代码和数据本身就是你的入场券。
