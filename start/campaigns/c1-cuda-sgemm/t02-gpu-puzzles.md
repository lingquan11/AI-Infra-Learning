# C1-T02 GPU-Puzzles 14 题

- **线**：必做 | **计划用时**：6h | **前置**：C1-T01（只需要掌握概念，不依赖本机环境）
- **一句话目标**：用 14 道小题把线程索引、边界保护、共享内存和同步练到不用思考，为 C1-T05 的共享内存分块打基础

## 材料（一手来源）
- 仓库：https://github.com/srush/GPU-Puzzles （按 README 用 Colab 打开 notebook，或按仓库说明在本地运行）
- 卡住时查阅：CUDA Programming Guide 中 Shared Memory 和 `__syncthreads` 的相关章节：https://docs.nvidia.com/cuda/cuda-programming-guide/

## 步骤
1. 把 notebook 另存到 `journey/exercises/gpu-puzzles/`（.ipynb，或导出为 .py）。
2. 按顺序做完 14 题（Map → … → Matrix Multiply）。每题先在纸上画出线程和数据的对应关系，再写代码。
3. 每题在 `journey/exercises/gpu-puzzles/notes.md` 中写一行：这题考什么、我第一次错在哪里（没有错就写"一次通过"）。
4. 网上有公开题解。**做完之前不看题解**；如果看了，在 notes.md 中这一题后面标【看过题解】，并在第二天闭卷重做一遍。

## 所有权
- `[me]`：全部题解和 notes.md
- `[ai]`：导师只按提示梯度给提示，不给可以直接粘贴的答案

## 机器验收（开工前写死，AI 不得修改）
- [ ] 14 题都运行了 notebook 自带的检查，并显示通过（输出保存在 notebook 中一起入库）
- [ ] `notes.md` 有 14 行，标记了哪些题看过题解；看过题解的题有重做记录

## 闭卷要点
- Guards 题：为什么线程数大于数据量时必须加判断？不加会怎样？
- Shared 题：共享内存的作用域和生命周期是什么？读写之间为什么需要 `__syncthreads()`？
- Dot Product / Prefix Sum：块内归约的步骤是怎样的？每一轮有多少线程是空闲的？
- Matrix Multiply（第 14 题）：分块的大小怎样决定共享内存的用量？它和 C1-T05 的 v3 有什么关系？
- 现场题：考官给一个新的 1D 模板计算（比如 3 点平均），口述线程映射和共享内存方案

## 卡住时
- 单题超过 45 分钟：允许看题解，按第 4 步标记，第二天重做。
- 总用时超过 9h：剩下的题直接看题解，标记后重做最后一题（Matrix Multiply）即可。
