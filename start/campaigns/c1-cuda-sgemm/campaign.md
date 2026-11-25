# 战役 C1：CUDA SGEMM 阶梯优化

- **周期**：2026-12-07 → 2027-01-10（截止日不延期；1 月考试周的预算按一半计）
- **预算**：每周 9–10h，合计约 48h（阅读 14h + 编程 29.5h + 动手 5.5h）
- **唯一交付物**：`results/sgemm/`：v1–v5 在 4096³ FP32 下的 GFLOPS 与占 cuBLAS 百分比的阶梯表，附 ncu 剖析
- **统一约定**：行主序 FP32，A、B 取 [-1, 1] 均匀分布、种子固定；与 cuBLAS 逐元素比较 `|C − C_ref| ≤ 1e-3 + 1e-3·|C_ref|`，必测 128、1000、4096；`cudaEvent` 计时，预热 3 次、正式 10 次取中位数；cuBLAS 用默认数学模式（不开 TF32）
- **推荐顺序**：R01 → H01 → P01 → R02 → R03 → P02 → R04 → R05 → H02 → P03 → R06 → P04 → P05 → R07 → P06 → P07 → R08 → H03 → H04（阅读按编号读即可，编号就是需要它的先后）

## 阅读（我自己读，不和 AI 对话；读完对 Muse 说"读完 <ID>"；状态：未读 / 在读 / 读完）
| ID | 长文 | 读哪些部分 | 预计用时 | 读完应该能回答（自测，不考核） | 状态 |
|---|---|---|---:|---|---|
| C1-R01 | Kirk, Hwu & El Hajj, [*Programming Massively Parallel Processors*（第 4 版）](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0) | 第 1–3 章：异构计算、数据并行、多维网格与数据 | 3h | grid / block / thread 怎样映射到数据？全局下标怎么算？为什么要边界检查？ | 未读 |
| C1-R02 | [PMPP 第 4 版](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0)（同上） | 第 4 章：计算架构与调度 | 2h | SM、warp、延迟隐藏、占用率分别是什么？GPU 为什么用更多线程而不是更大缓存来隐藏延迟？ | 未读 |
| C1-R03 | [PMPP 第 4 版](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0)（同上） | 第 5 章：存储架构与数据局部性 | 2.5h | 共享内存分块把全局内存访问减少到几分之一？和 CPU 缓存分块有何异同？ | 未读 |
| C1-R04 | [PMPP 第 4 版](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0)（同上） | 第 6 章：性能考量 | 2h | 合并访问的条件是什么？线程粗化为什么能提高性能？ | 未读 |
| C1-R05 | Horace He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) | 全文 | 1h | 计算受限、带宽受限、开销受限分别怎样判断？GFLOPS 和带宽利用率各说明什么？ | 未读 |
| C1-R06 | Simon Boehm, [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM) | 开头到 Kernel 3（Naive、全局内存合并、共享内存分块） | 1h | Naive 版慢在哪里？合并访问改了线程到元素的什么映射？共享内存分块减少了哪部分访存？ | 未读 |
| C1-R07 | Simon Boehm, [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM)（同上） | Kernel 4–5（1D / 2D Blocktiling）；Kernel 6 起留给加分 | 1h | 每个线程算多个结果为什么能提高算术强度？寄存器用量上升会带来什么代价？ | 未读 |
| C1-R08 | NVIDIA, [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | 指标结构、Speed of Light、Warp Stall Reasons 相关小节 | 1.5h | Compute SOL 和 Memory SOL 各衡量什么？哪些 stall reason 说明在等访存？ | 未读 |

没有 PMPP 纸质书时，R01–R04 用 [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 中编程模型、硬件实现、性能指南的对应部分代替（在阅读记录中注明）。

## 编程（GPT 带写，只看机器验收；状态：未开始 / 进行中 / 通过）
| ID | 任务 | 计划用时 | 验收要点（展开任务卡时不得弱于此） | 先读 | 依赖动手 | 状态 |
|---|---|---:|---|---|---|---|
| C1-P01 | 第一个 CUDA 程序：device-info + vector add | 2.5h | 见 [p01-first-kernel.md](p01-first-kernel.md) | C1-R01 | C1-H01 | 未开始 |
| C1-P02 | LeetGPU 热身 5 题 | 3h | 见 [p02-leetgpu.md](p02-leetgpu.md) | C1-R01–R03 | — | 未开始 |
| C1-P03 | 校验与计时框架 | 3h | cuBLAS 在 4096³ 下输出 GFLOPS；**故意写错的 kernel 能被判为 FAIL**；`[ai]` 可写样板，我能逐行解释 cuBLAS 行主序调用、误差判据和计时代码 | C1-R05 | C1-H02 | 未开始 |
| C1-P04 | v1 Naive + v2 全局内存合并访问 | 5h | 两版在三个规模下通过校验；4096 下的 GFLOPS 与占 cuBLAS 百分比写入 CSV | C1-R04、R06 | C1-H02 | 未开始 |
| C1-P05 | v3 共享内存分块 | 5h | 通过校验；4096 下比 v2 快（如果没有，在 C1-H03 中用 ncu 说明） | C1-R03、R06 | C1-H02 | 未开始 |
| C1-P06 | v4 1D Blocktiling | 5h | 通过校验；记录每线程结果数 TM 和寄存器数（`nvcc --resource-usage`） | C1-R07 | C1-H02 | 未开始 |
| C1-P07 | v5 2D Blocktiling | 6h | 通过校验；4096 下 ≥ cuBLAS 的 50%（未达到就在 C1-H03 中用 ncu 解释） | C1-R07 | C1-H02 | 未开始 |

## 动手（Muse 负责，只看机器验收；状态：未开始 / 进行中 / 完成）
| ID | 任务 | 计划用时 | 验收要点 | 先读 | 状态 |
|---|---|---:|---|---|---|
| C1-H01 | 环境：WSL2 + CUDA Toolkit + ncu | 1.5h | 见 [h01-cuda-wsl.md](h01-cuda-wsl.md) | — | 未开始 |
| C1-H02 | SGEMM 工程骨架 | 1h | 见 [h02-sgemm-skeleton.md](h02-sgemm-skeleton.md) | — | 未开始 |
| C1-H03 | 测量与剖析 | 2h | v1–v5 与 cuBLAS 的 GFLOPS 表（CSV + 图）；`ncu --set full` 采集 v1、v3、v5，报告或截图入库；`[me]` 由我写 3–5 句解读（Compute / Memory SOL 与 stall reasons 的变化） | C1-R05、R08 | 未开始 |
| C1-H04 | 收尾 | 1h | `results/sgemm/README.md` ≤ 30 行（阶梯表、图、边界）；打 tag `c1-v1.0`；ROADMAP 中 C1 改为完成 | — | 未开始 |

## 加分（必做全部完成后才碰）
| ID | 任务 | 计划用时 |
|---|---|---:|
| C1-B01 | v6：`float4` 向量化访存 + A 转置，目标 ≥ cuBLAS 的 70%（先读 siboehm Kernel 6） | 5h |
| C1-B02 | v7：消除 bank conflict + 双缓冲预取（先读 siboehm Kernel 7 起的相关部分） | 6h |

## 砍线顺序
1. C1-B02
2. C1-B01
3. （仍然超时）C1-P02 减为 3 题（每类一题）；C1-H03 只采集 v1 和 v5

## 收尾定义
2027-01-10：三栏必做全部完成 → 打 `c1-v1.0` → Muse 更新 ROADMAP 和时间线 → 转到 IC 仓库的寒假战役。没完成的必做项在 `results/sgemm/README.md` 的"边界"中写明，同样收尾。
