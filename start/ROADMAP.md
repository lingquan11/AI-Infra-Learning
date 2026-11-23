# 路线图：单卡计算与推理底层闭环

> 这是整条路线的唯一总表。当前进度看 [STATE.md](STATE.md)，当前战役在 [campaigns/](campaigns/) 中。
> 战役结束时只改本页的"状态"列，不改路线；路线调整须由我本人决定，并在本页末尾的"变更记录"中写一行。

## 原则
1. **一手来源 → 独立手写 → 硬件指标验收**：材料只用原仓库、原论文和官方文档；核心实现 `[me]` 由我自己写；验收看正确性和 profiler 数字，不看"学过"。
2. **战役制**：每个战役 3–5 周，只有一个公开交付物，分必做线和加分线，到期必须收尾（见 [RULES.md](RULES.md)）。
3. **每个战役单独有价值**：任何一个战役结束后停下来，仓库里都留有一份完整、可复现的成果。

## 前置与衔接
- **前置**：C0 在 [cpp-matmul-performance](https://github.com/lingquan11/cpp-matmul-performance) 仓库中进行（截止 2026-12-06）：CPU 上的访存、分块、SIMD 直觉
- **寒假**：转到 [IC-Design-Learning](https://github.com/lingquan11/IC-Design-Learning)（同样的仓库骨架和工作流）
- **寒假后**：回到本仓库继续 C2–C4

## 硬件与环境基线（C1-T01 实测核对，以实测为准）
- GPU：RTX 5060 Ti，Blackwell，计算能力 12.0（`sm_120`）
- 系统：Windows + WSL2（Ubuntu 24.04）；在 WSL2 内**只装 CUDA Toolkit，不装 Linux 显卡驱动**
- CUDA Toolkit ≥ 12.8；PyTorch ≥ 2.8（cu128 版本）；Nsight Compute（`ncu`）需要开启性能计数器权限

## 战役总表

| ID | 战役 | 窗口 | 预算 | 唯一交付物 | 必做验收摘要 | 状态 |
|---|---|---|---:|---|---|---|
| C0 | cpp-matmul 收尾 v1.0 | 2026-11-23 → 2026-12-06 | 16–20h | 在 cpp-matmul-performance 仓库 | 见该仓库 `start/` | 进行中 |
| C1 | CUDA SGEMM 阶梯优化 | 2026-12-07 → 2027-01-10 | 40–50h | `results/sgemm/`：v1–v5 + cuBLAS 基线 + ncu 报告 | v1–v5 全部通过数值校验；4096³ FP32 下 v5 ≥ cuBLAS 的 50%（未达到时用 ncu 数据解释差距）；ncu 对比 v1/v3/v5 | 未开始 |
| — | *寒假：IC 战役* | 约 2027-01 中旬 → 2 月中旬 | — | 见 IC-Design-Learning | — | — |
| C2 | Triton 与访存密集型融合算子 | 寒假后，暂定 4 周 | 30–40h | `results/triton-kernels/`：Fused Softmax、Fused RMSNorm + 带宽对比 | 两个算子 `allclose` 通过；多种隐藏层维度下的有效带宽（GB/s）对比 PyTorch eager 和 `torch.compile`，并计算占理论峰值的百分比 | 未开始 |
| C3 | FlashAttention-2 Forward 复现 | C2 之后，暂定 4–5 周 | 35–45h | `results/flash-attn-fwd/`：Triton 版 FA2 forward（支持 causal） | 与 `scaled_dot_product_attention` 对比 `allclose(atol=1e-2, rtol=1e-2)`；序列长度 512–4096 的延迟曲线和峰值显存曲线；手写 online softmax 推导 | 未开始 |
| C4 | 轻量推理引擎拆解 | C3 之后，暂定 3–4 周 | 25–35h | `results/nano-vllm-study/`：源码剖析 + 5060 Ti 实测报告 | Qwen3-0.6B 跑通 `example.py` 和 `bench.py`；手绘数据流图；二选一完成：A. 换用自己的 RMSNorm 后输出 token 一致；B. 消融实验，画出 TTFT/TPOT/吞吐曲线 | 未开始 |

合计约 130–170h。只有 C1 排在寒假前；C2–C4 的窗口在寒假结束、看到实际课表后再定。

---

## C1 CUDA SGEMM 阶梯优化
- **要回答的问题**：CPU 上学到的访存、分块和向量化直觉，在 GPU 的 SIMT 模型和存储层次（GMEM → L2 → SMEM → 寄存器）中分别对应什么？
- **锚点**：[srush/GPU-Puzzles](https://github.com/srush/GPU-Puzzles)（14 题热身）；Simon Boehm 的博客 [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM) 和仓库 [siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA)（只读，不 clone 运行）；*PMPP* 第 4 版第 3–6 章（按需查阅）；[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)；[Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/)
- **加分线候选**：v6 `float4` 向量化访存 + 转置（目标 ≥ 70% cuBLAS）；v7 消除 bank conflict + 双缓冲；warptiling
- **主要风险**：WSL2 下 `ncu` 拿不到硬件计数器 → C1-T01 第一周就验证，退路是改用 Windows 原生工具链

## C2 Triton 与融合算子
- **要回答的问题**：从 CUDA 的"单线程视角"切换到 Triton 的"block 张量视角"，访存密集型算子的上限由什么决定？
- **锚点**：[srush/Triton-Puzzles](https://github.com/srush/Triton-Puzzles)（前 8–10 题）；[Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html) 前 3 个（Vector Add、Fused Softmax、Matrix Multiplication）
- **必做**：Fused Softmax、Fused RMSNorm；带宽对比表；占峰值带宽的百分比
- **加分线候选**：LayerNorm backward；Triton matmul 与 C1 的 v5 对比
- **主要风险**：Triton 对 `sm_120` 的支持取决于 PyTorch 自带的 Triton 版本 → 第一张任务卡先跑通官方 vector add

## C3 FlashAttention-2 Forward
- **要回答的问题**：怎样在不物化 N×N 矩阵的前提下保持数值稳定？显存为什么能从 O(N²) 降到 O(N)？
- **锚点**：Tri Dao, *FlashAttention-2*（[arXiv:2307.08691](https://arxiv.org/abs/2307.08691)）第 2.3 节（FlashAttention 与 online softmax）、第 3.1 节（Algorithm 1）；FlashAttention-1（[arXiv:2205.14135](https://arxiv.org/abs/2205.14135)）作对照；Triton 官方教程 *Fused Attention*（只在自己写完后对照）
- **必做**：手写推导（拍照入库）→ 非 causal 版 → causal 版 → 精度验收 → 延迟和显存曲线
- **加分线候选**：backward pass；不同 head_dim 的性能对比

## C4 推理引擎拆解
- **要回答的问题**：KV Cache 怎样分页管理？调度器怎样在 prefill 和 decode 之间做决策？
- **锚点**：[GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)；vLLM 论文 *Efficient Memory Management for Large Language Model Serving with PagedAttention*（SOSP 2023，[arXiv:2309.06180](https://arxiv.org/abs/2309.06180)）前 4 节
- **必做**：跑通基准测试；逐个读 `LLMEngine`、`Scheduler`、`BlockManager`、`ModelRunner`，画出数据流图；完成选项 A 或 B
- **主要风险**：nano-vllm 依赖 `flash-attn`，它在 `sm_120` 上能否安装需要先验证 → 第一张任务卡只做环境验证，失败时记录原因并寻找替代方案

---

## 路线完成后的自检（C4 结束时做一次）
1. **独立手写**：给一个新的算子公式（比如某种 Normalization 变体），不参考现成实现，2 天内写出数值正确、带宽利用率合理的 Triton 或 CUDA 版本。
2. **剖析直觉**：拿到一段慢的推理代码，能用 `ncu` 或 PyTorch Profiler 判断它是算力受限、带宽受限，还是调度和 launch 开销受限。
3. **证据链**：GitHub 上有一条连续、可一键复现的链条：CPU 矩阵优化 → CUDA 阶梯优化 → Triton 论文复现 → 推理系统拆解。

之后的方向（FP8/INT4 量化、Tensor Core、CUTLASS/CuTe、向 vLLM/SGLang 提 PR）记在 [IDEAS.md](IDEAS.md)，C4 结束后再决定。

## 变更记录
- 2026-11-23：由 [原始 8 周手册](../journey/archive/2026-11-20-ai-infra-8-week-handbook.md) 改写。主要改动：时间按 1.5–2 倍重估，拆成 4 个战役；C++ 收尾移回 cpp-matmul 仓库；"v6 达到 cuBLAS 的 70%"从必做降为加分（必做改为 v5 ≥ 50% 或给出解释）；每个战役的第一张任务卡先排除环境风险。
