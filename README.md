# AI-Infra-Learning

From CUDA kernels to an LLM inference engine on a single RTX 5060 Ti: hand-written, profiled, reproducible.

在一张 RTX 5060 Ti 上，从 CUDA 算子一路做到 LLM 推理引擎：每一步都由我手写实现，用 profiler 验收，可以一键复现。

## 我怎么开始学习
1. 打开 [start/STATE.md](start/STATE.md)：当前战役、当前任务、下一步
2. 打开它指向的任务卡
3. 把 [start/prompts/tutor.md](start/prompts/tutor.md) + STATE + 任务卡贴给导师 AI，开始学习
4. 学完用 [start/prompts/archiver.md](start/prompts/archiver.md) 归档；任务做完用 `examiner.md` 考核

路线见 [start/ROADMAP.md](start/ROADMAP.md)（C1 SGEMM → C2 Triton → C3 FlashAttention-2 → C4 nano-vllm），规则见 [start/RULES.md](start/RULES.md)。

## 学习过程
[journey/](journey/README.md)：时间线、每次学习的日志、笔记、考核记录。

## 学习成果
| 做了什么 | 结论 | 在哪里 |
|---|---|---|
| — | C1 从 2026-12-07 开始，完成后在这里放 SGEMM 阶梯表 | [results/](results/README.md) |

相关仓库：[cpp-matmul-performance](https://github.com/lingquan11/cpp-matmul-performance)（前置 C0：CPU 矩阵乘法）· [IC-Design-Learning](https://github.com/lingquan11/IC-Design-Learning)（数字 IC 路线）
