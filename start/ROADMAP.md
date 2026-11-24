# 路线图：从 CPU 矩阵乘法到单卡推理与分布式训练

> 这是整条路线的唯一总表。当前进度看 [STATE.md](STATE.md)，当前战役在 [campaigns/](campaigns/) 中，每个任务的阅读材料在 [reading-map.md](reading-map.md)。
> 战役结束时只改本页的"状态"列；路线调整由我本人决定，并在本页末尾的"变更记录"中写一行。

## 起点：我现在会什么（依据 cpp-matmul 仓库的证据和我的自述）
| 已有 | 还没有（由各战役的"起步任务"按需补上） |
|---|---|
| C++ 基础（数组、循环、计时）；**指针与动态内存能读懂基本代码**；Windows + MSVC | Linux / WSL、CMake → C1-T01 |
| 矩阵乘法、循环顺序、分块；正确性校验与中位数测量 | CUDA（从零开始）→ C1 |
| CPU 缓存行、带宽、IPC、Top-down 的概念（Day 8）；**Python + NumPy 能读懂基本代码** | PyTorch → C2-T01 |
| Git、GitHub | Transformer 与注意力 → C3-T01；LLM 推理 → C4-T01；集合通信 → C5-T01 |

**原则：前置知识只在用到之前补，不集中补课。** 每个战役的第一个任务是"起步任务"：先做一个短诊断，已经会的部分直接跳过，只补缺口。

## 贯穿全程的五条线索（为什么是这个顺序）
| 线索 | C0 cpp | C1 CUDA | C2 Triton | C3 注意力 | C4 推理引擎 | C5 分布式 |
|---|---|---|---|---|---|---|
| **矩阵乘法** | 循环顺序、分块 | 共享内存 / 寄存器分块 | Triton matmul，对照 C1 | 注意力 = 两次矩阵乘 + softmax | 模型前向中的 GEMM | 张量并行切分矩阵乘 |
| **归约与 softmax** | — | LeetGPU 的归约题 | 融合 softmax | online softmax → FA2 | 采样中的 softmax | all-reduce 就是跨卡归约 |
| **访存与显存** | CPU 缓存 | 合并访问、共享内存 | 带宽利用率 | 显存从 O(N²) 降到 O(N) | KV cache 分页管理 | ZeRO 分片 |
| **测量** | `chrono` 中位数 | `cudaEvent` + ncu | `triton.testing` + 带宽百分比 | PyTorch profiler / nsys | TTFT / TPOT / 吞吐 | 通信量与扩展性 |
| **成果复用** | — | v5 SGEMM | C2 的 matmul 对照 C1 | 自己的 FA2 kernel | **用自己的 FA2 替换 flash-attn** | 在自己的小模型上做并行 |

## 硬件与环境（C1-T02 实测核对）
- RTX 5060 Ti，Blackwell，计算能力 12.0（`sm_120`）；Windows + WSL2（Ubuntu 24.04），WSL 内只装 CUDA Toolkit，不装 Linux 显卡驱动
- CUDA Toolkit ≥ 12.8；PyTorch ≥ 2.7 的 cu128 版本（C2-T01 安装）；ncu 需要在 Windows 端开启性能计数器

## 战役总表
| ID | 战役 | 窗口 | 预算 | 唯一交付物 | 必做验收摘要 | 状态 |
|---|---|---|---:|---|---|---|
| C0 | cpp-matmul 收尾 v1.0 | 2026-11-23 → 2026-12-06 | 16–20h | 在 cpp-matmul-performance 仓库 | 见该仓库 `start/` | 进行中 |
| C1 | CUDA SGEMM 阶梯优化 | 2026-12-07 → 2027-01-10 | 40–50h | `results/sgemm/`：v1–v5 + cuBLAS 基线 + ncu 报告 | v1–v5 全部通过数值校验；4096³ FP32 下 v5 ≥ cuBLAS 的 50%（未达到时用 ncu 数据解释）；ncu 对比 v1/v3/v5 | 未开始 |
| — | *寒假：IC 战役* | 约 2027-01 中旬 → 2 月中旬 | — | 见 IC-Design-Learning | — | — |
| C2 | Triton 与融合算子 | 寒假后，约 4 周 | 25–35h | `results/triton-kernels/`：融合 softmax、融合 RMSNorm + 带宽对比 | 两个算子 `allclose` 通过；多种隐藏层维度下的带宽对比 PyTorch eager 和 `torch.compile`，并算出占峰值的百分比 | 未开始 |
| C3 | FlashAttention-2 Forward | C2 之后，约 4–5 周 | 35–45h | `results/flash-attn-fwd/`：Triton 版 FA2 forward（支持 causal） | 通过 CS336 A2 的 FA2 测试；与 `scaled_dot_product_attention` 对比 `allclose(atol=1e-2, rtol=1e-2)`；序列长度 512–4096 的延迟和显存曲线 | 未开始 |
| C4 | 推理引擎拆解 | C3 之后，约 3–4 周 | 25–35h | `results/nano-vllm-study/`：源码剖析 + 换上自己的 FA2 的实测 | Qwen3-0.6B 跑通 `example.py` / `bench.py`；手绘数据流图；用自己的 FA2（或 Triton attention）替换 flash-attn 后输出一致，并对比吞吐 | 未开始 |
| C5 | 分布式训练（单机模拟） | C4 之后，约 4 周 | 30–40h | `results/distributed-from-scratch/`：DDP、ZeRO-1、TP、PP 的从零实现 | 用 gloo 多进程（CPU）运行，各并行方式的 loss 与单进程训练一致；手推显存与通信量 | 未开始 |

合计约 170–220h。只有 C1 排在寒假前；C2–C5 的具体窗口在寒假后看课表再定。

---

## C1 CUDA SGEMM 阶梯优化
- **起步任务 C1-T01**：诊断（指针只做快速确认：cudaMalloc 和 kernel 参数都是指针，要能写出来，不只是读懂）+ WSL / Linux + CMake
- **热身**：[LeetGPU](https://leetgpu.com/challenges)：在浏览器里写 CUDA C++，跑在真实 GPU 上（CUDA 12.8），不用担心本地环境
- **主线**：Simon Boehm 的博客 [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)（Kernel 1–5）；*PMPP* 第 4 版第 2–6 章（按需查阅）
- **对照代码**：[LeetCUDA](https://github.com/xlite-dev/LeetCUDA) 的 `kernels/sgemm`、`kernels/reduce`（自己写完再看）
- **加分**：v6 `float4` 向量化（≥ 70% cuBLAS）、v7 双缓冲、warptiling
- **风险**：WSL2 下 ncu 拿不到计数器 → C1-T02 验证，退路是改用 Windows 原生工具链

## C2 Triton 与融合算子
- **起步任务 C2-T01**：PyTorch 张量基础（从 NumPy 迁移：设备、dtype、`contiguous`、GPU 计时）→ 安装 PyTorch（cu128）→ 在 `sm_120` 上跑通 Triton 官方的 vector add（同时验证 Triton 支持）
- **主线**：[Triton-Puzzles](https://github.com/gpu-mode/Triton-Puzzles)（前 8–10 题）；[Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html) 前 3 个
- **衔接**：融合 softmax 是 C3 online softmax 的前一步；Triton matmul 与 C1 的 v5 对比
- **加分**：LayerNorm backward

## C3 FlashAttention-2 Forward
- **起步任务 C3-T01**：用 PyTorch 手写一个注意力层（跟着 Karpathy 的 [Let's build GPT](https://github.com/karpathy/ng-video-lecture)）→ 手算注意力的 FLOPs 和显存 → 数值稳定的 softmax
- **主线**：Stanford [CS336 Assignment 2](https://github.com/stanford-cs336/assignment2-systems) 第 4 节（FA2 Triton kernel，作业自带测试）；*FlashAttention-2*（[arXiv:2307.08691](https://arxiv.org/abs/2307.08691)）第 2.3 节和 Algorithm 1
- **加分**：backward pass；不同 head_dim 的性能对比

## C4 推理引擎拆解
- **起步任务 C4-T01**：LLM 推理基础（prefill / decode、KV cache 大小手算）→ 用 HF transformers 跑一次生成 → **验证 nano-vllm 依赖的 flash-attn 能否在 `sm_120` 上安装**（社区报告目前不支持；装不上就直接进入"替换"路线）
- **主线**：[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)；vLLM 论文（[arXiv:2309.06180](https://arxiv.org/abs/2309.06180)）前 4 节；AIInfraGuide 推理模块第 3 章"nano-vllm 源码解读"作为中文导读
- **加分**：KV cache 量化；消融实验（CUDA Graph 开关、并发数）

## C5 分布式训练（单机模拟）
- **起步任务 C5-T01**：集合通信原语（broadcast / reduce / all-reduce / all-gather / reduce-scatter）→ 用 `torch.distributed` + gloo 在单机起多进程跑通 all-reduce
- **主线**：CS336 A2 的 DDP 与优化器状态分片部分（作业说明原文：本地开发用 Gloo，可以在只有 CPU 的机器上运行）；HF [picotron_tutorial](https://github.com/huggingface/picotron_tutorial)（分步视频 + 代码；[picotron](https://github.com/huggingface/picotron) 支持 `--use_cpu` 在 CPU 上跑 3D 并行）
- **单卡做不到的**：真实互联带宽和多卡加速比
- **加分**：租 2–4 张卡几小时，跑 `nccl-tests` 并对比 gloo 的结果

---

## 路线完成后的自检（C5 结束时）
1. **独立手写**：给一个新算子公式，2 天内写出数值正确、带宽利用率合理的 Triton 或 CUDA 版本。
2. **剖析直觉**：拿到一段慢代码，能用 ncu / PyTorch Profiler 判断是算力受限、带宽受限，还是调度和 launch 开销受限。
3. **证据链**：GitHub 上有一条连续、可复现的链条：CPU 矩阵乘 → CUDA 阶梯优化 → Triton 融合 → FA2 → 推理引擎改造 → 分布式从零实现。

## 变更记录
- 2026-11-23：由 [原始 8 周手册](../journey/archive/2026-11-20-ai-infra-8-week-handbook.md) 改写为 C1–C4。
- 2026-11-24：按入门性与连贯性重排：每个战役加"起步任务"（前置按需补）；C1 热身由 GPU-Puzzles（Python，Colab 有报告损坏）换成 LeetGPU（CUDA C++）；PyTorch 安装移到 C2-T01；C3 以 CS336 A2 为主线；C4 改为用自己的 FA2 替换 flash-attn（flash-attn 在 sm_120 上没有官方支持）；新增 C5 分布式训练（单机 gloo 模拟）；新增五条贯穿线索和导读表。
- 2026-11-24：按自述更新起点：指针与动态内存、Python + NumPy 已有基础；C1-T01 缩为诊断 + Linux + CMake（2h），C2-T01 只补 PyTorch。
