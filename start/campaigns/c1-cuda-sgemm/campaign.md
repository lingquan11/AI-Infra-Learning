# 战役 C1：CUDA SGEMM 阶梯优化

- **周期**：2026-12-07 → 2027-01-10（截止日不延期）
- **预算**：每周 8–10h，合计 40–50h；必做线（学习）计划 38h，另有环境任务约 2–3h（与 Muse 协作）。1 月考试周的预算按一半计，超时就按砍线顺序处理
- **唯一交付物**：`results/sgemm/`。一张表展示 v1–v5 在 4096³ FP32 下的 GFLOPS 和占 cuBLAS 的百分比，附 ncu 剖析报告
- **前置**：C0（cpp-matmul v1.0）已收尾；工具链由环境任务 C1-S01、C1-S02 准备（可以在 12-07 之前和 Muse 提前做）
- **和 C0 的衔接**：C0 中"分块在小规模不加速、缓存决定性能"的结论，在 GPU 上对应"合并访问 → 共享内存分块 → 寄存器分块"三步

## 统一约定（所有任务共用）
- 矩阵按行主序存储，FP32，C = A·B，A、B 的元素取 [-1, 1] 均匀分布，随机种子固定
- **数值校验**：与 cuBLAS 结果逐元素比较，要求 `|C − C_ref| ≤ 1e-3 + 1e-3·|C_ref|`。必测规模：128、1000（不是 32 的倍数）、4096
- **计时**：`cudaEvent`，先预热 3 次，再正式计时 10 次取中位数。GFLOPS = 2·M·N·K / 时间
- **cuBLAS 基线**：`cublasSgemm`，默认数学模式（确认没有开启 TF32）
- **编译**：CMake，`CMAKE_CUDA_ARCHITECTURES 120`，Release
- **结果**：每个版本的 CSV 写入 `results/sgemm/experiments/<版本>/data/`，记录 GPU 型号、驱动版本、CUDA 版本和 git 提交号

## 必做线（状态：未开始 / 进行中 / 待考官 / 通过）
| ID | 任务 | 计划用时 | 验收要点（展开任务卡时不得弱于此） | 状态 |
|---|---|---:|---|---|
| C1-T01 | 起步：指针快速确认 + GPU 硬件与编程模型 | 1.5h | 见 [t01-start.md](t01-start.md) | 未开始 |
| C1-T02 | 第一个 CUDA 程序：device-info + Vector Add + 第一次 ncu | 2.5h | 见 [t02-first-kernel.md](t02-first-kernel.md) | 未开始 |
| C1-T03 | LeetGPU 热身 8 题 | 5h | 在 LeetGPU 上通过 8 题，必须包含向量运算、矩阵乘、归约各至少 1 题；每题在 `journey/exercises/leetgpu/` 存代码和一行笔记；闭卷：线程索引、边界判断、共享内存与 `__syncthreads()`、归约的步骤 | 未开始 |
| C1-T04 | 校验与计时：cuBLAS 基线、误差判据、cudaEvent 计时、CSV | 3h | cuBLAS 在 4096³ 下输出 GFLOPS；**故意写错的 kernel 能被校验函数判为 FAIL**；`[ai]` 可以写辅助代码，但我必须能逐行解释 cuBLAS 行主序调用、误差判据和计时代码 | 未开始 |
| C1-T05 | v1 Naive + v2 全局内存合并访问 | 5h | 两个版本在三个规模下都通过校验；4096 下的 GFLOPS 和占 cuBLAS 百分比写入 CSV；闭卷：画出一个 warp 内 32 个线程在 v1 和 v2 中访问的地址模式 | 未开始 |
| C1-T06 | v3 共享内存分块 | 5h | 通过校验；4096 下比 v2 快（如果没有，就用 ncu 数据说明原因）；闭卷：两处 `__syncthreads()` 分别在防止什么；与 C0 的 CPU 分块做对照 | 未开始 |
| C1-T07 | v4 1D Blocktiling（每个线程计算多个结果） | 5h | 通过校验；记录 TM 和每线程寄存器数（`nvcc --resource-usage` 或 ncu）；闭卷：用公式说明算术强度如何提高 | 未开始 |
| C1-T08 | v5 2D Blocktiling（寄存器级分块） | 6h | 通过校验；4096 下 **≥ cuBLAS 的 50%**，未达到就在 C1-T09 中用 ncu 解释差距；记录寄存器数和占用率 | 未开始 |
| C1-T09 | ncu 剖析 + 报告 + 收尾 | 5h | 用 `ncu --set full` 采集 v1、v3、v5，报告文件入库（太大就入库截图）；`results/sgemm/README.md` 中解释 Compute SOL、Memory SOL、Warp Stall Reasons 的变化；`results/sgemm/summary.md` ≤ 800 字；打 tag `c1-v1.0` | 未开始 |

C1-T03 到 C1-T09 的任务卡在前一张通过后再展开，文件名按内容命名（如 `t03-leetgpu.md`），见 [RULES.md](../../RULES.md) §2。

## 环境准备（与 Muse 协作，按 [setup.md](../../prompts/setup.md)；状态：未开始 / 进行中 / 就绪）
| ID | 内容 | 验收命令 / 要点 | 被哪些学习任务依赖 | 状态 |
|---|---|---|---|---|
| C1-S01 | Windows 驱动与性能计数器权限；WSL2（Ubuntu 24.04）；CUDA Toolkit ≥ 12.8（WSL-Ubuntu 包）；g++、CMake、git | 见 [s01-cuda-wsl.md](s01-cuda-wsl.md) | C1-T02、T05–T09 | 未开始 |
| C1-S02 | SGEMM 工程骨架：CMake（`CMAKE_CUDA_ARCHITECTURES 120`、Release）、链接 cuBLAS、`results/sgemm/` 目录结构、一键构建运行脚本 | `cmake --build` 成功；骨架程序调用一次 `cublasSgemm` 并正常退出；新增一个 kernel 版本只需加一个文件 | C1-T04 起 | 未开始 |

## 加分线（必做线全部通过后才碰）
| ID | 任务 | 计划用时 |
|---|---|---:|
| C1-B01 | v6：`float4` 向量化访存 + A 转置，目标 ≥ cuBLAS 的 70% | 5h |
| C1-B02 | v7：消除 bank conflict + 双缓冲预取 | 6h |
| C1-B03 | warptiling（对应 siboehm 的 kernel 10） | 8h |

## 砍线顺序
1. C1-B03
2. C1-B02
3. C1-B01
4. （仍然超时）C1-T03 减为 5 题；C1-T09 只采集 v1 和 v5 两份 ncu 报告

## 收尾定义
2027-01-10：必做线全部通过 → 打 `c1-v1.0` → 在 ROADMAP 中更新状态 → 在 `journey/README.md` 的时间线中加一行 → 转到 IC 仓库的寒假战役。
有必做项没有完成：在 summary 的"未完成"一节写明原因，同样收尾。
