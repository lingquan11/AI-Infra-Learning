# 学习成果

暂无。C1 CUDA SGEMM 从 2026-12-07 开始，计划目录：

| 目录 | 内容 | 战役 |
|---|---|---|
| `env-check/`、`hw/` | 环境自检程序、RTX 5060 Ti 硬件事实（工具读数） | C1-T01 |
| `sgemm/` | v1–v5 对比 cuBLAS 的阶梯表 + ncu 剖析 | C1 |
| `triton-kernels/` | Fused Softmax / RMSNorm + 带宽对比 | C2 |
| `flash-attn-fwd/` | FlashAttention-2 forward（Triton） | C3 |
| `nano-vllm-study/` | 源码剖析 + 实测报告 | C4 |

每个战役完成后，这里换成结论表（格式见 [start/templates/results-readme.md](../start/templates/results-readme.md)）。
