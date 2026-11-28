# 学习成果

<!-- results:start -->
暂无成果：C1 CUDA SGEMM 从 2026-12-07 开始，第一个任务通过后这里换成结论表。
<!-- results:end -->

计划目录：

| 目录 | 内容 | 战役 |
|---|---|---|
| `env-check/`、`hw/` | 环境自检程序、RTX 5060 Ti 硬件事实（工具读数） | C1-H01 / C1-P01 |
| `sgemm/` | v1–v5 对比 cuBLAS 的阶梯表 + ncu 剖析 | C1 |
| `triton-kernels/` | Fused Softmax / RMSNorm + 带宽对比 | C2 |
| `flash-attn-fwd/` | FlashAttention-2 forward（Triton） | C3 |
| `nano-vllm-study/` | 源码剖析 + 换上自己的 FA2 的实测 | C4 |
| `distributed-from-scratch/` | DDP、ZeRO-1、TP、PP 的从零实现（单机 gloo） | C5 |

结论表格式见 [start/templates/results-readme.md](../start/templates/results-readme.md)。
