# 路线图：从 CUDA 算子到单卡推理与分布式训练

> 这是整条路线的唯一总表。当前进度看 [STATE.md](STATE.md)，当前战役在 [campaigns/](campaigns/) 中。
> 战役结束时只改本页的"状态"列；路线调整由我本人决定，并在本页末尾的"变更记录"中写一行。

## 三栏学习
| 栏 | 谁 | 内容 |
|---|---|---|
| **阅读** `Rxx` | 我自己，不和 AI 对话 | 主教材的章节：C1–C4 全部用 [AIInfraGuide](https://caomaolufei.github.io/AIInfraGuide/)（中文开源教程，从 CUDA 到推理一条线），C5 用 Hugging Face 的 Ultra-Scale Playbook；每篇附自测问题，不考核；读完告诉 Muse 记录 |
| **编程** `Pxx` | GPT 带我写 | kernel 与算子本体由我写，GPT 只按提示梯度引导；机器验收 |
| **动手** `Hxx` | Muse | 环境、工具链、测量与剖析、作图、整理结果；需要解读的部分由我写 |

**主线**：每个战役先读主教材、再写、再测，编程和动手任务都标明"先读"哪几篇，三栏计入进度。**支线**：与主教材配套的现成实践课程，选做、不计入进度，读完对应章节后做。

| 战役 | 主线主教材 | 支线实践课程 |
|---|---|---|
| C1 | AIInfraGuide：GPU 基础、CUDA 第 1、2、4 章 | [LeetGPU](https://leetgpu.com/challenges) 的 CUDA 题 |
| C2 | AIInfraGuide：CUDA 第 3、5、7 章 | LeetGPU 用 Triton 做同类题（归约、softmax） |
| C3 | AIInfraGuide：Transformer、FlashAttention | LeetGPU 的 softmax / attention 题 |
| C4 | AIInfraGuide：推理第 1–3 章 | 无：编程任务本身就是读改 nano-vllm 源码 |
| C5 | Ultra-Scale Playbook | 无：picotron_tutorial 已是主线的配套代码 |

## 起点（依据 cpp-matmul 仓库的证据和我的自述）
- 已有：C++（含指针与动态内存）、Python + NumPy 能读懂基本代码；矩阵乘法的循环顺序与分块；CPU 缓存、带宽、IPC 的概念；Windows + MSVC
- 还没有：CUDA、PyTorch、Transformer、Linux 工具链（工具链由 Muse 在各战役的第一个动手任务中准备）

## 贯穿全程的线索
| 线索 | C1 CUDA | C2 Triton | C3 注意力 | C4 推理引擎 | C5 分布式 |
|---|---|---|---|---|---|
| 矩阵乘法 | 共享内存 / 寄存器分块 | Triton matmul，对照 C1 | 注意力 = 两次矩阵乘 + softmax | 模型前向中的 GEMM | 张量并行切分矩阵乘 |
| 归约与 softmax | 共享内存与同步 | 归约、融合 softmax | online softmax → FA2 | 采样 | all-reduce 就是跨卡归约 |
| 访存与显存 | 合并访问、共享内存 | 带宽利用率 | 显存 O(N²) → O(N) | KV cache 分页 | ZeRO 分片 |
| 成果复用 | v5 SGEMM | 融合算子 | 自己的 FA2 kernel | **用自己的 FA2 替换 flash-attn** | 在自己的小模型上做并行 |

## 战役总表
| ID | 战役 | 窗口 | 预算 | 唯一交付物 | 必做验收摘要 | 状态 |
|---|---|---|---:|---|---|---|
| C0 | cpp-matmul 收尾 v1.0 | 2026-11-23 → 2026-12-06 | 16–20h | 在 cpp-matmul-performance 仓库 | 见该仓库 `start/` | 进行中 |
| C1 | CUDA SGEMM 阶梯优化 | 2026-12-07 → 2027-01-10 | 40–45h | `results/sgemm/`：v1–v5 + cuBLAS 基线 + ncu 剖析 | 读完 AIInfraGuide 的 GPU 基础与 CUDA 第 1、2、4 章；v1–v5 全部通过数值校验，v5 ≥ cuBLAS 的 50%（未达到时用 ncu 解释） | 未开始 |
| — | *寒假：IC 战役* | 约 2027-01 中旬 → 2 月中旬 | — | 见 IC-Design-Learning | — | — |
| C2 | Triton 与融合算子 | 寒假后，约 4 周 | 25–35h | `results/triton-kernels/` | 融合 softmax、融合 RMSNorm 通过 `allclose`；带宽对比表与占峰值百分比 | 未开始 |
| C3 | FlashAttention-2 Forward | C2 之后，约 5 周 | 35–45h | `results/flash-attn-fwd/` | 通过 CS336 A2 的 FA2 测试；512–4096 的延迟与显存曲线 | 未开始 |
| C4 | 推理引擎拆解 | C3 之后，约 4 周 | 25–35h | `results/nano-vllm-study/` | Qwen3-0.6B 跑通；用自己的 FA2 替换 flash-attn 后输出一致，并对比吞吐 | 未开始 |
| C5 | 分布式训练（单机模拟） | C4 之后，约 4 周 | 30–40h | `results/distributed-from-scratch/` | DDP、ZeRO-1、TP、PP 的 loss 与单进程训练一致（gloo 多进程） | 未开始 |

合计约 180–220h。只有 C1 排在寒假前；C2–C5 的窗口在寒假后看课表再定。

---

## C1 CUDA SGEMM 阶梯优化
- **阅读**（AIInfraGuide）：前置 · [GPU 基础知识](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/gpu/gpu-basics)；CUDA · [第 1 章 编程入门](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/第1章-cuda编程入门)、[第 2 章 性能优化基础](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/第2章-cuda性能优化基础)、[第 4 章 GEMM](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/41-cuda-gemm算子性能优化)
- **编程**：vector add → 校验与计时框架 → v1–v5 SGEMM
- **动手**：WSL2 + CUDA + ncu → SGEMM 工程骨架 → v1–v5 测量与 ncu 剖析 → 收尾
- **风险**：WSL2 下 ncu 拿不到计数器 → C1-H01 验证，退路是 Windows 原生工具链

## C2 Triton 与融合算子
- **阅读**（AIInfraGuide）：CUDA · [第 3 章 Reduce](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/31-cuda-reduce算子优化)、第 5 章 [Softmax 朴素实现](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/51-cuda-softmax朴素实现优化)与 [Online Softmax](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/52-cuda-online-softmax实现)、[第 7 章 Triton](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/71-triton编程框架)；[Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html) 01–03 在编程任务中照着写，不单列阅读
- **编程**：改写 Triton vector add → 融合 softmax → 融合 RMSNorm；加分：Triton matmul 对照 C1 v5
- **动手**：安装 PyTorch（cu128）与 Triton，验证 `sm_120`；带宽基准与作图

## C3 FlashAttention-2 Forward
- **阅读**（AIInfraGuide）：前置 · 第 3 章 Transformer 的 [3.2 全貌及代码实现](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/transformer/32-transformer全貌及代码实现)、[3.3 Self-Attention](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/transformer/33-self-attention机制深入理解)、[3.8 从 Transformer 到 LLM 自回归生成](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/transformer/38-从transformer到llm自回归生成深入理解)；CUDA · 第 6 章 [FlashAttention V1](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/61-flashattention-v1详解)、[FlashAttention V2](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/62-flashattention-v2详解)
- **编程**：PyTorch 手写注意力 → Triton FA2 forward（以 [CS336 A2](https://github.com/stanford-cs336/assignment2-systems) 的测试为验收）→ causal
- **动手**：CS336 A2 环境；延迟与峰值显存曲线

## C4 推理引擎拆解
- **阅读**（AIInfraGuide）：推理 · [1.1 LLM 推理基础](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第1章-llm推理基础/11-llm推理基础)；第 2 章的 [2.1 PagedAttention](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/21-pagedattention)、[2.2 Continuous Batching](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/22-continuous-batching)；[第 3 章 nano-vllm 源码解读](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/第3章-nano-vllm源码解读) 3.1–3.6
- **编程**：用自己的 FA2 替换 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的 flash-attn（varlen prefill 与 paged decode）
- **动手**：nano-vllm 环境、Qwen3-0.6B、flash-attn 在 `sm_120` 上能否安装的结论；基准与消融实验作图

## C5 分布式训练（单机模拟）
- **阅读**（主教材例外：AIInfraGuide 分布式模块的 ZeRO、TP、PP 等章节还没写，C5 整个换成一本书）：Hugging Face, [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) 全书（单卡训练显存 → DP → ZeRO → TP / SP → PP → 选配置）；它的配套代码 [picotron_tutorial](https://github.com/huggingface/picotron_tutorial) 在编程任务中对照
- **编程**：从零实现 DDP（梯度分桶）、ZeRO-1、TP、PP，参照 [picotron_tutorial](https://github.com/huggingface/picotron_tutorial)
- **动手**：单机 gloo 多进程环境；记录各并行方式的显存与通信量；加分：租卡跑 `nccl-tests`

---

## 变更记录
- 2026-11-23：由原始 8 周手册改写为战役制（见 `journey/archive/`）。
- 2026-11-24：按入门性重排，新增 C5。
- 2026-11-25：改为三栏学习：理论改为自己读长文、不和 AI 对话、不考核；编程由 GPT 带写；动手（环境、测量、剖析）由 Muse 负责。原导师带学版本见 [journey/archive/2026-11-24-tutor-led/](../journey/archive/2026-11-24-tutor-led/)。
- 2026-11-25：阅读改为"一个战役一份主教材"：C1–C4 用 AIInfraGuide，C5 用 Ultra-Scale Playbook，不再在 PMPP、博客和论文之间切换。
