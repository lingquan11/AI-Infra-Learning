# 路线图：从 CUDA 算子到单卡推理与分布式训练

> 这是整条路线的唯一总表。当前进度看 [STATE.md](STATE.md)，当前战役在 [campaigns/](campaigns/) 中。
> 战役结束时只改本页的"状态"列；路线调整由我本人决定，并在本页末尾的"变更记录"中写一行。

## 三栏学习
| 栏 | 谁 | 内容 |
|---|---|---|
| **阅读** `Rxx` | 我自己，不和 AI 对话 | 教材、经典论文、作者本人写的长文；每篇附自测问题，不考核；读完告诉 Muse 记录 |
| **编程** `Pxx` | GPT 带我写 | kernel 与算子本体由我写，GPT 只按提示梯度引导；机器验收 |
| **动手** `Hxx` | Muse | 环境、工具链、测量与剖析、作图、整理结果；需要解读的部分由我写 |

每个战役先读、再写、再测：编程和动手任务都标明"先读"哪几篇。

## 起点（依据 cpp-matmul 仓库的证据和我的自述）
- 已有：C++（含指针与动态内存）、Python + NumPy 能读懂基本代码；矩阵乘法的循环顺序与分块；CPU 缓存、带宽、IPC 的概念；Windows + MSVC
- 还没有：CUDA、PyTorch、Transformer、Linux 工具链（工具链由 Muse 在各战役的第一个动手任务中准备）

## 贯穿全程的线索
| 线索 | C1 CUDA | C2 Triton | C3 注意力 | C4 推理引擎 | C5 分布式 |
|---|---|---|---|---|---|
| 矩阵乘法 | 共享内存 / 寄存器分块 | Triton matmul，对照 C1 | 注意力 = 两次矩阵乘 + softmax | 模型前向中的 GEMM | 张量并行切分矩阵乘 |
| 归约与 softmax | LeetGPU 的归约题 | 融合 softmax | online softmax → FA2 | 采样 | all-reduce 就是跨卡归约 |
| 访存与显存 | 合并访问、共享内存 | 带宽利用率 | 显存 O(N²) → O(N) | KV cache 分页 | ZeRO 分片 |
| 成果复用 | v5 SGEMM | 融合算子 | 自己的 FA2 kernel | **用自己的 FA2 替换 flash-attn** | 在自己的小模型上做并行 |

## 战役总表
| ID | 战役 | 窗口 | 预算 | 唯一交付物 | 必做验收摘要 | 状态 |
|---|---|---|---:|---|---|---|
| C0 | cpp-matmul 收尾 v1.0 | 2026-11-23 → 2026-12-06 | 16–20h | 在 cpp-matmul-performance 仓库 | 见该仓库 `start/` | 进行中 |
| C1 | CUDA SGEMM 阶梯优化 | 2026-12-07 → 2027-01-10 | 45–50h | `results/sgemm/`：v1–v5 + cuBLAS 基线 + ncu 剖析 | 读完 PMPP 第 1–6 章与 siboehm 长文；v1–v5 全部通过数值校验，v5 ≥ cuBLAS 的 50%（未达到时用 ncu 解释） | 未开始 |
| — | *寒假：IC 战役* | 约 2027-01 中旬 → 2 月中旬 | — | 见 IC-Design-Learning | — | — |
| C2 | Triton 与融合算子 | 寒假后，约 4 周 | 25–35h | `results/triton-kernels/` | 融合 softmax、融合 RMSNorm 通过 `allclose`；带宽对比表与占峰值百分比 | 未开始 |
| C3 | FlashAttention-2 Forward | C2 之后，约 5 周 | 35–45h | `results/flash-attn-fwd/` | 通过 CS336 A2 的 FA2 测试；512–4096 的延迟与显存曲线 | 未开始 |
| C4 | 推理引擎拆解 | C3 之后，约 4 周 | 25–35h | `results/nano-vllm-study/` | Qwen3-0.6B 跑通；用自己的 FA2 替换 flash-attn 后输出一致，并对比吞吐 | 未开始 |
| C5 | 分布式训练（单机模拟） | C4 之后，约 4 周 | 30–40h | `results/distributed-from-scratch/` | DDP、ZeRO-1、TP、PP 的 loss 与单进程训练一致（gloo 多进程） | 未开始 |

合计约 180–220h。只有 C1 排在寒假前；C2–C5 的窗口在寒假后看课表再定。

---

## C1 CUDA SGEMM 阶梯优化
- **阅读**（按此顺序）：Kirk & Hwu, [*Programming Massively Parallel Processors*（第 4 版）](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0)第 1–6 章；Horace He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)；Simon Boehm, [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)（分两段，配合 v1–v5 读）；[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- **编程**：vector add → LeetGPU 热身 → 校验与计时框架 → v1–v5 SGEMM
- **动手**：WSL2 + CUDA + ncu → SGEMM 工程骨架 → v1–v5 测量与 ncu 剖析 → 收尾
- **风险**：WSL2 下 ncu 拿不到计数器 → C1-H01 验证，退路是 Windows 原生工具链

## C2 Triton 与融合算子
- **阅读**（按此顺序）：Kirk & Hwu, [PMPP](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0) 第 10 章（归约与减少分支发散，softmax 的基础）；Tillet, Kung, Cox, *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*（[MAPL 2019](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)）；[Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html) 01–03；Milakov & Gimelshein, *Online normalizer calculation for softmax*（[arXiv:1805.02867](https://arxiv.org/abs/1805.02867)）
- **编程**：改写 Triton vector add → 融合 softmax → 融合 RMSNorm；加分：Triton matmul 对照 C1 v5
- **动手**：安装 PyTorch（cu128）与 Triton，验证 `sm_120`；带宽基准与作图

## C3 FlashAttention-2 Forward
- **阅读**（按此顺序）：Harvard NLP, [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)（注意力部分）；Zihao Ye, [From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)；Dao et al., *FlashAttention*（[arXiv:2205.14135](https://arxiv.org/abs/2205.14135)）第 1–3 节；Dao, *FlashAttention-2*（[arXiv:2307.08691](https://arxiv.org/abs/2307.08691)）第 2–3 节
- **编程**：PyTorch 手写注意力 → Triton FA2 forward（以 [CS336 A2](https://github.com/stanford-cs336/assignment2-systems) 的测试为验收）→ causal
- **动手**：CS336 A2 环境；延迟与峰值显存曲线

## C4 推理引擎拆解
- **阅读**（按此顺序）：Lilian Weng, [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)（全景）；Anyscale, [How continuous batching enables 23x throughput](https://www.anyscale.com/blog/continuous-batching-llm-inference)；Kwon et al., *PagedAttention / vLLM*（[arXiv:2309.06180](https://arxiv.org/abs/2309.06180)）第 1–4 节；[AIInfraGuide › 推理](https://caomaolufei.github.io/AIInfraGuide/inference/) 第 3 章 nano-vllm 源码解读（中文）
- **编程**：用自己的 FA2 替换 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的 flash-attn（varlen prefill 与 paged decode）
- **动手**：nano-vllm 环境、Qwen3-0.6B、flash-attn 在 `sm_120` 上能否安装的结论；基准与消融实验作图

## C5 分布式训练（单机模拟）
- **阅读**（按此顺序）：Hugging Face, [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)；Rajbhandari et al., *ZeRO*（[arXiv:1910.02054](https://arxiv.org/abs/1910.02054)）；Shoeybi et al., *Megatron-LM*（[arXiv:1909.08053](https://arxiv.org/abs/1909.08053)）；选读 [How To Scale Your Model](https://jax-ml.github.io/scaling-book/)
- **编程**：从零实现 DDP（梯度分桶）、ZeRO-1、TP、PP，参照 [picotron_tutorial](https://github.com/huggingface/picotron_tutorial)
- **动手**：单机 gloo 多进程环境；记录各并行方式的显存与通信量；加分：租卡跑 `nccl-tests`

---

## 变更记录
- 2026-11-23：由原始 8 周手册改写为战役制（见 `journey/archive/`）。
- 2026-11-24：按入门性重排，新增 C5。
- 2026-11-25：改为三栏学习：理论改为自己读长文、不和 AI 对话、不考核；编程由 GPT 带写；动手（环境、测量、剖析）由 Muse 负责。原导师带学版本见 [journey/archive/2026-11-24-tutor-led/](../journey/archive/2026-11-24-tutor-led/)。
