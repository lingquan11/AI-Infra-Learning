# 战役 C1：CUDA SGEMM 阶梯优化

- **周期**：2026-12-07 → 2027-01-10（截止日不延期；1 月考试周的预算按一半计）
- **预算**：每周 9–10h，合计约 40.5h（阅读 8.5h + 编程 26.5h + 动手 5.5h）；支线另计
- **唯一交付物**：`results/sgemm/`：v1–v5 在 4096³ FP32 下的 GFLOPS 与占 cuBLAS 百分比的阶梯表，附 ncu 剖析
- **主线**：读 AIInfraGuide → 编程 P01–P06（vector add → 校验与计时 → v1–v5）→ 动手 H01–H04（三张表，计入进度）
- **支线**（选做，不计入进度）：[LeetGPU](https://leetgpu.com/challenges)，在线 CUDA 题库（AIInfraGuide 推荐的练习平台），浏览器里写、真实 GPU 上判题，和主线读到的章节对应
- **主教材**：[AIInfraGuide](https://caomaolufei.github.io/AIInfraGuide/)（中文开源教程）的"GPU 硬件概论"与"CUDA 编程与算子优化"模块；阅读表全部出自它。它的第 8 章（性能分析工具链）还没写，C1-H03 解读 ncu 时查阅 NVIDIA [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- **统一约定**：行主序 FP32，A、B 取 [-1, 1] 均匀分布、种子固定；与 cuBLAS 逐元素比较 `|C − C_ref| ≤ 1e-3 + 1e-3·|C_ref|`，必测 128、1000、4096；`cudaEvent` 计时，预热 3 次、正式 10 次取中位数；cuBLAS 用默认数学模式（不开 TF32）
- **推荐顺序**：R01 → H01 → R02 → P01 → R03 → R04 → H02 → R05 → P02 → P03 → P04 → R06 → P05 → P06 → H03 → H04（阅读按编号读即可，编号就是需要它的先后）

## 阅读（我自己读，不和 AI 对话；读完对 Muse 说"读完 <ID>"；状态：未读 / 在读 / 读完）
| ID | 长文 | 读哪些部分 | 预计用时 | 读完应该能回答（自测，不考核） | 状态 |
|---|---|---|---:|---|---|
| C1-R01 | AIInfraGuide 前置 · 第 5 章 [GPU 基础知识：从硬件架构到 AI 计算](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/gpu/gpu-basics) | 全文 | 1h | GPU 和 CPU 在设计上做了什么取舍？SM、warp、Tensor Core 分别是什么？显存带宽和算力谁更容易成为瓶颈？ | 未读 |
| C1-R02 | AIInfraGuide CUDA · [第 1 章 CUDA 编程入门](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/第1章-cuda编程入门) | [1.2 编程模型](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/12-cuda编程模型)、[1.3 内存模型](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/13-cuda内存模型)、[1.4 第一个实用 Kernel](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/14-第一个实用kernel)（1.1 环境搭建交给 Muse） | 2h | grid / block / thread 怎样映射到数据？全局下标怎么算？为什么要边界检查？全局内存、共享内存、寄存器各有多快？ | 未读 |
| C1-R03 | AIInfraGuide CUDA · [第 2 章 CUDA 性能优化基础](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/第2章-cuda性能优化基础) | [2.1 Warp 与执行模型](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/21-warp与执行模型)、[2.2 内存访问优化](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/22-内存访问优化) | 1.5h | warp 分歧怎样产生？合并访问的条件是什么？共享内存的 bank conflict 怎样避免？ | 未读 |
| C1-R04 | AIInfraGuide CUDA · [第 2 章 CUDA 性能优化基础](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/第2章-cuda性能优化基础)（同上） | [2.3 Occupancy 与资源分配](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/23-occupancy与资源分配)、[2.4 同步与原子操作](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/24-同步与原子操作) | 1.5h | 占用率受哪几种资源限制？`__syncthreads()` 的语义和常见陷阱？原子操作为什么慢、怎样减少冲突？ | 未读 |
| C1-R05 | AIInfraGuide CUDA · 第 4 章 [4.1 CUDA GEMM 算子性能优化](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/41-cuda-gemm算子性能优化) | 第 1–4 节：背景与性能指标、Roofline 与计算访存比、朴素实现、Thread Block 级 Tiling | 1.5h | 计算访存比怎么算？block tiling 把全局内存访问减少到几分之一？为什么沿 K 维切分？ | 未读 |
| C1-R06 | AIInfraGuide CUDA · [4.1 CUDA GEMM 算子性能优化](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/41-cuda-gemm算子性能优化)（同上） | 第 5 节：Thread 级 Tiling（外积分解、寄存器预算）；第 6–9 节留给加分 | 1h | 每个线程算多个结果为什么能提高寄存器复用？寄存器用量上升会带来什么代价？ | 未读 |

## 编程（GPT 带写，只看机器验收；状态：未开始 / 进行中 / 通过）
| ID | 任务 | 计划用时 | 验收要点（展开任务卡时不得弱于此） | 先读 | 依赖动手 | 状态 |
|---|---|---:|---|---|---|---|
| C1-P01 | 第一个 CUDA 程序：device-info + vector add | 2.5h | 见 [p01-first-kernel.md](p01-first-kernel.md) | C1-R01、R02 | C1-H01 | 未开始 |
| C1-P02 | 校验与计时框架 | 3h | cuBLAS 在 4096³ 下输出 GFLOPS；**故意写错的 kernel 能被判为 FAIL**；`[ai]` 可写样板，我能逐行解释 cuBLAS 行主序调用、误差判据和计时代码 | C1-R05 | C1-H02 | 未开始 |
| C1-P03 | v1 Naive + v2 全局内存合并访问 | 5h | 两版在三个规模下通过校验；4096 下的 GFLOPS 与占 cuBLAS 百分比写入 CSV | C1-R03、R05 | C1-H02 | 未开始 |
| C1-P04 | v3 共享内存分块 | 5h | 通过校验；4096 下比 v2 快（如果没有，在 C1-H03 中用 ncu 说明） | C1-R05 | C1-H02 | 未开始 |
| C1-P05 | v4 1D Blocktiling | 5h | 通过校验；记录每线程结果数 TM 和寄存器数（`nvcc --resource-usage`） | C1-R06 | C1-H02 | 未开始 |
| C1-P06 | v5 2D Blocktiling | 6h | 通过校验；4096 下 ≥ cuBLAS 的 50%（未达到就在 C1-H03 中用 ncu 解释） | C1-R06 | C1-H02 | 未开始 |

## 动手（Muse 负责，只看机器验收；状态：未开始 / 进行中 / 完成）
| ID | 任务 | 计划用时 | 验收要点 | 先读 | 状态 |
|---|---|---:|---|---|---|
| C1-H01 | 环境：WSL2 + CUDA Toolkit + ncu | 1.5h | 见 [h01-cuda-wsl.md](h01-cuda-wsl.md) | — | 未开始 |
| C1-H02 | SGEMM 工程骨架 | 1h | 见 [h02-sgemm-skeleton.md](h02-sgemm-skeleton.md) | — | 未开始 |
| C1-H03 | 测量与剖析 | 2h | v1–v5 与 cuBLAS 的 GFLOPS 表（CSV + 图）；`ncu --set full` 采集 v1、v3、v5，报告或截图入库；`[me]` 由我写 3–5 句解读（指标含义查阅 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)）（Compute / Memory SOL 与 stall reasons 的变化） | C1-R03、R06 | 未开始 |
| C1-H04 | 收尾 | 1h | `results/sgemm/README.md` ≤ 30 行（阶梯表、图、边界）；打 tag `c1-v1.0`；ROADMAP 中 C1 改为完成 | — | 未开始 |

## 支线（现成实践课程，选做，不计入进度；做完对 Muse 说"做完 <ID>"；状态：未开始 / 完成）
| ID | 实践课程 | 做哪些 | 配合主线（读完 / 做完之后） | 预计用时 | 状态 |
|---|---|---|---|---:|---|
| C1-S01 | [LeetGPU](https://leetgpu.com/challenges) | 2 道逐元素题（如向量加、ReLU）：练线程索引和边界判断 | C1-R02 | 1h | 未开始 |
| C1-S02 | LeetGPU（同上） | 1 道用共享内存和 `__syncthreads()` 的题（如矩阵转置）+ 1 道朴素矩阵乘 | C1-R04 | 2h | 未开始 |
| C1-S03 | LeetGPU（同上） | 1 道归约题（如求和），为 C2 的 softmax 预热 | C1-R04 | 1h | 未开始 |

题解网上都有，做完之前不看；想留档就把代码存到 `journey/exercises/leetgpu/`。

## 加分（必做全部完成后才碰）
| ID | 任务 | 计划用时 |
|---|---|---:|
| C1-B01 | v6：`float4` 向量化访存 + A 转置，目标 ≥ cuBLAS 的 70%（先读 C1-R05 同篇第 7 节） | 5h |
| C1-B02 | v7：消除 bank conflict + 双缓冲预取（先读 C1-R05 同篇第 8–9 节） | 6h |

## 砍线顺序
1. C1-B02
2. C1-B01
3. （仍然超时）C1-H03 只采集 v1 和 v5

## 收尾定义
2027-01-10：三栏必做全部完成 → 打 `c1-v1.0` → Muse 更新 ROADMAP 和时间线 → 转到 IC 仓库的寒假战役。没完成的必做项在 `results/sgemm/README.md` 的"边界"中写明，同样收尾。
