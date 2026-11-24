# STATE（≤30 行，只写当前）

- 更新：2026-11-23
- 战役：C1 CUDA SGEMM 阶梯优化（2026-12-07 → 2027-01-10）→ [campaigns/c1-cuda-sgemm/campaign.md](campaigns/c1-cuda-sgemm/campaign.md)
- 当前任务：C1-T01 起步：指针确认 + GPU 心智模型（2026-12-07 开始，等 C0 收尾）→ [t01-start.md](campaigns/c1-cuda-sgemm/t01-start.md)
- 本战役已用：0h / 预算 40–50h

## 上次停在哪
- 仓库刚建立，还没有开始学习

## 未解决的问题
- WSL2 下 ncu 能不能拿到硬件计数器？（C1-T01 验证，失败就改用 Windows 原生工具链）
- RTX 5060 Ti 的显存是 8GB 还是 16GB？（C1-T01 读取，会影响 C3/C4 的规模上限）

## 下一步
- 12-07 之前（可选）：和 Muse 完成环境任务 C1-S01（WSL + CUDA + ncu）→ [s01-cuda-wsl.md](campaigns/c1-cuda-sgemm/s01-cuda-wsl.md)
- 2026-12-07 起：C1-T01（不依赖环境）

## 环境
- 待 C1-S01 填写：Windows 版本、驱动版本、WSL 发行版、CUDA 版本（详见 `results/hw/environment.md`）
