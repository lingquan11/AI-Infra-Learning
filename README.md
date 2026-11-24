# AI-Infra-Learning

From CUDA kernels to an LLM inference engine on a single RTX 5060 Ti: hand-written, profiled, reproducible.

在一张 RTX 5060 Ti 上，从 CUDA 算子一路做到 LLM 推理引擎：每一步都由我手写实现，用 profiler 验收，可以一键复现。

## 学习进度
<!-- progress:start -->
```
当前战役 C1 CUDA SGEMM 阶梯优化
  ░░░░░░░░░░░░░░░░░░░░   0%  0/9 个必做任务通过 · 已用 0h / 计划 40h · 截止 2027-01-10
整条路线 C0 → C5
  ░░░░░░░░░░░░░░░░░░░░   0%  0/6 个战役完成
当前任务 C1-T01 起步：诊断 + Linux + CMake（2026-12-07 开始，等 C0 收尾） · 最近学习 2026-11-23
```
<!-- progress:end -->

## 我怎么开始学习
在 Codex 中打开本仓库，说"开始学习"；学完说"结束"，Codex 自动归档并更新进度。流程见 [AGENTS.md](AGENTS.md)；路线见 [start/ROADMAP.md](start/ROADMAP.md)（C1 CUDA → C2 Triton → C3 FlashAttention-2 → C4 推理引擎 → C5 分布式训练），每个任务的阅读材料见 [start/reading-map.md](start/reading-map.md)。

## 学习过程
[journey/](journey/README.md)：时间线、每次学习的日志、笔记、考核记录。

## 学习成果
<!-- results:start -->
暂无成果：C1 CUDA SGEMM 从 2026-12-07 开始，第一个任务通过后这里换成结论表。
<!-- results:end -->

相关仓库：[cpp-matmul-performance](https://github.com/lingquan11/cpp-matmul-performance)（前置 C0：CPU 矩阵乘法）· [IC-Design-Learning](https://github.com/lingquan11/IC-Design-Learning)（数字 IC 路线）
