# 导读表

导师在进入任务、或讲到新概念之前，从这里挑**最多 2 篇**给我（规则见 `start/prompts/tutor.md`）。路线由战役和任务卡决定，导读只服务于当前任务。
读过的条目在日志的"做了什么"里写编号（如 `读 R-C1-05`），导师不再重复推荐。

**时机**：做前 = 动手前读，建立概念；卡住 = 遇到对应问题时读；做后 = 自己写完再读，用来对照。
**常用来源**：[AIInfraGuide](https://caomaolufei.github.io/AIInfraGuide/)（中文，下表写"Guide"）· [siboehm 的 SGEMM 博客](https://siboehm.com/articles/22/CUDA-MMM) · *PMPP* 第 4 版 · [LeetCUDA](https://github.com/xlite-dev/LeetCUDA) · 论文和官方文档

## C1 CUDA SGEMM（逐任务）
| 编号 | 任务 | 时机 | 来源与章节 | 读完要回答 | 时长 |
|---|---|---|---|---|---|
| R-C1-01 | T01 | 卡住（第 1、2 题 < 2 分时） | [learncpp.com](https://www.learncpp.com/)：Pointers、Dynamic memory allocation | `new[]` 和 `delete[]` 必须配对的原因？指针 +1 移动了多少字节？ | 60m |
| R-C1-02 | T01 | 卡住 | [Missing Semester：The Shell](https://missing.csail.mit.edu/2020/course-shell/) | 绝对路径和相对路径；管道 `\|` 做了什么？ | 45m |
| R-C1-03 | T02 | 做前 | [Guide › 前置知识](https://caomaolufei.github.io/AIInfraGuide/prerequisites/) 第 5 章「GPU 硬件概论」+ 5.1 架构演进 | SM、warp、显存层次分别对应 CPU 的什么？为什么 GPU 用更多线程来隐藏延迟？ | 40m |
| R-C1-04 | T02 | 做前 | [Guide › CUDA](https://caomaolufei.github.io/AIInfraGuide/cuda/) 1.2 编程模型、1.3 内存模型、1.4 向量加法 | grid / block / thread 怎样算出全局下标？host 和 device 内存怎样拷贝？ | 50m |
| R-C1-05 | T02 | 做后 | *PMPP* 第 2 章（Heterogeneous data parallel computing） | 与 Guide 对照：错误检查为什么要放在 kernel 启动之后？ | 40m |
| R-C1-06 | T03 | 做前 | Guide › CUDA 2.1 warp、2.4 同步与原子操作 | `__syncthreads()` 同步的范围是什么？atomicAdd 为什么慢？ | 40m |
| R-C1-07 | T03 | 卡住 | Guide › CUDA 3.1 Reduce；*PMPP* 第 10 章 Reduction | 树形归约每一轮有多少线程空闲？怎样减少分支发散？ | 50m |
| R-C1-08 | T04 | 做前 | siboehm 博客开头至 Kernel 1 之前；[cuBLAS 文档](https://docs.nvidia.com/cuda/cublas/) `cublasSgemm` | cuBLAS 是列主序：怎样用它算行主序的 C = A·B？ | 40m |
| R-C1-09 | T05 | 做前 | siboehm Kernel 1–2；Guide › CUDA 2.2 内存访问 | 合并访问的条件是什么？v1 中一个 warp 读 B 时访问了多少个缓存行？ | 50m |
| R-C1-10 | T06 | 做前 | siboehm Kernel 3；*PMPP* 第 5 章（Memory architecture and data locality） | 共享内存分块把全局内存访问减少到原来的几分之一？和 C0 的 CPU 分块有何相同、有何不同？ | 60m |
| R-C1-11 | T07 | 做前 | siboehm Kernel 4；Guide › CUDA 2.3 占用率 | 每个线程算更多结果，为什么能提高算术强度？寄存器用多了会怎样影响占用率？ | 50m |
| R-C1-12 | T08 | 做前 | siboehm Kernel 5；Guide › CUDA 4.1 GEMM | 二维寄存器分块中，每次从共享内存读的数据被复用了多少次？ | 60m |
| R-C1-13 | T08 | 做后 | LeetCUDA `kernels/sgemm` | 它和你的 v5 有哪些不同？哪一处是你下一步最值得借鉴的？ | 40m |
| R-C1-14 | T09 | 做前 | [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：Speed of Light、Warp Stall Reasons | Compute SOL 和 Memory SOL 分别衡量什么？哪个 stall reason 说明在等访存？ | 50m |

## C2 Triton 与融合算子（任务卡展开时细化）
| 编号 | 任务 | 时机 | 来源与章节 | 读完要回答 |
|---|---|---|---|---|
| R-C2-01 | 起步 | 做前 | [PyTorch 60 Minute Blitz](https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)：Tensors 一节；Guide › 前置知识 第 4 章 PyTorch | 和 NumPy 数组相比，PyTorch 张量多了什么（设备、自动求导）？`.contiguous()` 是什么意思？为什么 GPU 计时前要 `torch.cuda.synchronize()`？ |
| R-C2-02 | 起步 | 做前 | [Triton 教程](https://triton-lang.org/main/getting-started/tutorials/index.html) 01 Vector Addition | Triton 的一个 program 对应 CUDA 的什么？`BLOCK_SIZE` 和 mask 的作用？ |
| R-C2-03 | 融合 softmax | 做前 | Triton 教程 02 Fused Softmax；Guide › CUDA 5.1 朴素 softmax | 融合之后省掉了哪几次显存读写？ |
| R-C2-04 | 融合 RMSNorm | 做后 | LeetCUDA `kernels/rms-norm`、`kernels/openai-triton` | 与你的实现对比：带宽利用率差在哪里？ |
| R-C2-05 | Triton matmul | 做前 | Triton 教程 03 Matrix Multiplication | 它的分块和你 C1 的 v5 分别是怎样对应的？ |

## C3 FlashAttention-2 Forward（任务卡展开时细化）
| 编号 | 任务 | 时机 | 来源与章节 | 读完要回答 |
|---|---|---|---|---|
| R-C3-01 | 起步 | 做前 | Karpathy [Let's build GPT](https://github.com/karpathy/ng-video-lecture)（自注意力部分）；Guide › 前置知识 3.3 自注意力 | Q、K、V 的形状？注意力矩阵为什么是 N×N？ |
| R-C3-02 | 起步 | 做前 | Guide › CUDA 5.2 Online Softmax | 怎样在只扫一遍的情况下算出数值稳定的 softmax？ |
| R-C3-03 | FA2 推导 | 做前 | *FlashAttention-2*（[arXiv:2307.08691](https://arxiv.org/abs/2307.08691)）第 2.3 节、Algorithm 1；Guide › CUDA 6.1–6.2 | 为什么显存从 O(N²) 降到了 O(N)？分块时 rescale 的因子是什么？ |
| R-C3-04 | FA2 实现 | 做前 | [CS336 A2](https://github.com/stanford-cs336/assignment2-systems) 作业说明第 4 节 | 作业要求的接口、测试和性能指标是什么？ |
| R-C3-05 | FA2 实现 | 做后 | LeetCUDA `kernels/flash-attn` | 它的分块参数和你的有什么不同？ |

## C4 推理引擎拆解（任务卡展开时细化）
| 编号 | 任务 | 时机 | 来源与章节 | 读完要回答 |
|---|---|---|---|---|
| R-C4-01 | 起步 | 做前 | [Guide › 推理](https://caomaolufei.github.io/AIInfraGuide/inference/) 1.1 推理基础 | prefill 和 decode 的瓶颈分别是什么？Qwen3-0.6B 的 KV cache 每个 token 占多少字节？ |
| R-C4-02 | 源码精读 | 做前 | Guide › 推理 2.1 PagedAttention、2.2 Continuous Batching；vLLM 论文（[arXiv:2309.06180](https://arxiv.org/abs/2309.06180)）第 1–4 节 | 逻辑块和物理块怎样映射？为什么能减少显存碎片？ |
| R-C4-03 | 源码精读 | 卡住 | Guide › 推理 3.1–3.6 nano-vllm 源码解读 | `Scheduler` 每一步怎样决定做 prefill 还是 decode？ |
| R-C4-04 | 替换 attention | 做前 | nano-vllm 的 [`nanovllm/layers/attention.py`](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/layers/attention.py)（调用 `flash_attn_varlen_func` 和 `flash_attn_with_kvcache`） | 这两个函数分别用在 prefill 还是 decode？要替换它们，你的 FA2 需要支持哪些输入格式？ |

## C5 分布式训练（任务卡展开时细化）
| 编号 | 任务 | 时机 | 来源与章节 | 读完要回答 |
|---|---|---|---|---|
| R-C5-01 | 起步 | 做前 | [Guide › 前置知识](https://caomaolufei.github.io/AIInfraGuide/prerequisites/) 第 6 章 集合通信；[PyTorch 分布式教程](https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html) | all-reduce 可以拆成哪两个原语？ring all-reduce 的通信量是多少？ |
| R-C5-02 | DDP | 做前 | CS336 A2 的 DDP 部分；[Guide › 分布式](https://caomaolufei.github.io/AIInfraGuide/distributed/) 4.1–4.2 | 梯度分桶为什么能让通信和计算重叠？ |
| R-C5-03 | ZeRO | 做前 | ZeRO 论文（[arXiv:1910.02054](https://arxiv.org/abs/1910.02054)）第 5 节；Guide › 分布式 1.1 显存估算 | ZeRO-1 / 2 / 3 各切分了什么？显存分别省多少？ |
| R-C5-04 | TP / PP | 做前 | [picotron_tutorial](https://github.com/huggingface/picotron_tutorial) 对应章节的视频；Guide › 分布式 第 6、7 章 | 张量并行在 MLP 中怎样切分才能只做一次 all-reduce？1F1B 为什么能减少显存？ |
