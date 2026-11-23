# STATE（≤30 行，只写当前）

- 更新：2026-11-23
- 战役：C1 CUDA SGEMM 阶梯优化（2026-12-07 → 2027-01-10）→ [campaigns/c1-cuda-sgemm/campaign.md](campaigns/c1-cuda-sgemm/campaign.md)
- 当前任务：C1-T01 环境打通（2026-12-07 开始，等 C0 收尾）→ [t01-env-setup.md](campaigns/c1-cuda-sgemm/t01-env-setup.md)
- 本战役已用：0h / 预算 40–50h

## 上次停在哪
- 仓库刚建立，还没有开始学习

## 未解决的问题
- WSL2 下 ncu 能不能拿到硬件计数器？（C1-T01 验证，失败就改用 Windows 原生工具链）
- RTX 5060 Ti 的显存是 8GB 还是 16GB？（C1-T01 读取，会影响 C3/C4 的规模上限）

## 下一步
- 2026-12-07 起：按 C1-T01 安装 WSL2 和 CUDA Toolkit

## 环境
- 待 C1-T01 填写：Windows 版本、驱动版本、WSL 发行版、CUDA 版本、PyTorch 版本
