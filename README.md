# AI-Infra-Learning

From CUDA kernels to an LLM inference engine on a single RTX 5060 Ti: hand-written, profiled, reproducible.

在一张 RTX 5060 Ti 上，从 CUDA 算子一路做到 LLM 推理引擎：每一步都由我手写实现，用 profiler 验收，可以一键复现。

## 学习进度
<!-- progress:start -->
```
当前战役 C1 CUDA SGEMM 阶梯优化 · 截止 2027-01-10
  阅读 ░░░░░░░░░░░░░░░░░░░░   0%  0/7 项完成 · 已用 0h / 计划 14h
  编程 ░░░░░░░░░░░░░░░░░░░░   0%  0/7 项完成 · 已用 0h / 计划 29.5h
  动手 ░░░░░░░░░░░░░░░░░░░░   0%  0/4 项完成 · 已用 0h / 计划 5.5h
整条路线 C0 → C5
  ░░░░░░░░░░░░░░░░░░░░   0%  0/6 个战役完成
当前阅读 C1-R01 PMPP 第 1–3 章（可以现在就开始读）
当前编程任务 C1-P01 第一个 CUDA 程序（依赖 C1-H01）
当前动手任务 C1-H01 WSL2 + CUDA + ncu（可以提前做）
最近学习 2026-11-25
```
<!-- progress:end -->

## 我怎么开始学习
| 我想 | 怎么做 |
|---|---|
| 学理论 | 读当前战役[阅读表](start/campaigns/c1-cuda-sgemm/campaign.md)里的长文（自己读，不找 AI）；读完对 Muse 说"读完 C1-R01" |
| 写代码 | 对 GPT 说"开始写代码" … "结束" |
| 装环境、跑实验 | 对 Muse 说"准备环境" / "做 C1-H01" |
| 看整条路线 | [ROADMAP](start/ROADMAP.md)：C1 CUDA → C2 Triton → C3 FA2 → C4 推理引擎 → C5 分布式 |

## 学习过程
[journey/](journey/README.md)：时间线、编程日志（GPT）、动手与阅读日志（Muse）、我自己的笔记。

## 学习成果
<!-- results:start -->
暂无成果：C1 CUDA SGEMM 从 2026-12-07 开始，第一个任务通过后这里换成结论表。
<!-- results:end -->

相关仓库：[cpp-matmul-performance](https://github.com/lingquan11/cpp-matmul-performance)（前置 C0：CPU 矩阵乘法）· [IC-Design-Learning](https://github.com/lingquan11/IC-Design-Learning)（数字 IC 路线） · 工作流：[learning-workflow](https://github.com/lingquan11/learning-workflow)（三个仓库共用的规则、提示词和进度脚本）
