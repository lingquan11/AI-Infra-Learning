# C1-T01 起步：诊断 + WSL / Linux + CMake

- **线**：必做 | **计划用时**：2h（诊断全部满分时约 0.5h） | **前置**：C0 收尾
- **一句话目标**：补齐写 CUDA 之前的两样工具（Linux 和 CMake），并确认指针已经能**写**出来：`cudaMalloc` 和 kernel 参数都是指针

## 第一步：诊断（30 分钟，不看资料、不问 AI）
在 `journey/exercises/c1-start/diagnosis.md` 中作答，每题按 0 / 1 / 2 分自评（0 = 不会，1 = 能写但不确定，2 = 能写也能解释）：
1. `int* p = new int[n];` 之后，`p[3]`、`*(p + 3)`、`&p[3]` 分别是什么？用完之后要做什么？
2. 一个 n×n 矩阵按行主序存放在一维数组里，元素 (i, j) 的下标是多少？怎样把它交给一个接收 `float*` 的函数？
3. 在 Linux 终端里：进入目录、列出文件、创建目录、查看文件内容、搜索文件中的文字，分别用什么命令？
4. 用 `g++` 编译一个 `.cpp` 并运行，命令是什么？加上 `-O2` 和 `-g` 分别有什么作用？
5. 一个最小的 `CMakeLists.txt` 至少要写哪几行？`cmake -B build` 和 `cmake --build build` 分别做什么？

**得 2 分的题，跳过下面对应的补课部分。** 你已有指针基础，第 1、2 题预计直接满分，重点在第 3–5 题。

## 第二步：只补得分 < 2 的部分
| 题号 | 补课材料（一手） | 练习 |
|---|---|---|
| 1、2 | [learncpp.com](https://www.learncpp.com/) 中 Pointers、Dynamic memory allocation、`std::vector` 的章节 | 【me】`journey/exercises/c1-start/matrix.cpp`：用 `new` 分配 n×n 矩阵，按 `i*n + j` 存放，写一个 `void matmul(const float* A, const float* B, float* C, int n)`，与 C0 的 ikj 结果对比；最后改用 `std::vector<float>` 加 `.data()` 再写一遍 |
| 3 | [MIT Missing Semester：The Shell](https://missing.csail.mit.edu/2020/course-shell/) | 在 WSL 中完成该讲的练习，把用到的命令记进 `notes.md` |
| 4、5 | [CMake 官方教程](https://cmake.org/cmake/help/latest/guide/tutorial/index.html) 第 1 步 | 【me】给 `matrix.cpp` 写 `CMakeLists.txt`，在 WSL 中构建运行 |

（WSL2 和 Ubuntu 的安装放在 C1-T02 第 2 步；如果第 3 题要在 Linux 中练习，就先做那一步。）

## 所有权
- `[me]`：诊断作答、`matrix.cpp`、`CMakeLists.txt`
- `[ai]`：可以写 `check.sh`（编译并用 AddressSanitizer 运行）

## 机器验收（开工前写死，AI 不得修改）
- [ ] `diagnosis.md` 中 5 题都有作答和自评分
- [ ] 第 1、2 题有低于 2 分的：在 WSL 中 `g++ -fsanitize=address -g matrix.cpp` 运行无内存错误，结果与 ikj 一致（n = 1、7、64）
- [ ] 第 4、5 题有低于 2 分的：`cmake -B build && cmake --build build` 成功
- [ ] 补课后重新自评，5 题都达到 2 分（达不到的写进 STATE 的"未解决的问题"，不阻塞 C1-T02）

## 闭卷要点（考官据此出题）
- 指针运算与数组下标的等价关系；悬空指针、内存泄漏分别怎么产生
- 行主序下标公式；把 `std::vector` 交给接收裸指针的函数
- 为什么 GPU 编程离不开指针（预告：`cudaMalloc` 返回的是设备内存的地址）

## 卡住时
- 超过 3h：先保证第 3 题（Linux 命令）达到 2 分，CMake 在 C1-T04 搭工程框架时边用边学。
