# IDEAS（停车场：想到就记一行，战役结束再看）

## C5 之后的方向（原手册的 Stage 2，届时再决定是否写成战役）
- FP8 / INT4 权重量化：利用 Blackwell 原生支持的低精度格式
- Tensor Core 入门：WMMA → CuTe → CUTLASS，与 C1 的 SIMT 版 SGEMM 对比
- 精读 vLLM 或 SGLang 的一个核心模块，尝试提交第一个 Bug Fix 或性能优化 PR

## 与 IC 线的交叉
- C1 的分块 GEMM 与脉动阵列的数据复用对照：同一个 GEMM，GPU 的 SMEM/寄存器分块和 NPU 的 PE 阵列分别是怎样复用数据的？

## 备选练习（不在主线中）
- [GPU-Puzzles](https://github.com/srush/GPU-Puzzles)（Python + Numba，Colab 版本有损坏报告）：学完 C2 的 Python 后可以作为趣味练习
- [gpu-mode/reference-kernels](https://github.com/gpu-mode/reference-kernels)：GPU MODE 排行榜题目（含 PMPP 系列），C1 之后可以挑战
- AIInfraGuide 中暂不使用的部分：推测解码、分布式推理、P/D 分离、生产部署（C4/C5 之后再看）
