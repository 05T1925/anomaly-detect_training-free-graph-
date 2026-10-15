# EBGAD 冒烟测试记录

最后更新：2026-10-12

## 论文与代码

- 论文：*Graph Anomaly Detection as Finite-Horizon Control: Training-Free Scoring via Empirical Bayes*，arXiv:2609.38424v1，2026-09-29。
- 本地论文：Papers/methods/EB-GAD-arXiv-2609.38424.pdf。
- 作者仓库：heraclixus/EBGAD；本地 commit：cba2969e390ae6cd4381fe86393fa6cfcfc40d13。
- 仓库 README 仍写旧标题 *Graph Anomaly Detection as Dynamical Transport*；本地 PDF 是 Finite-Horizon Control 标题，保留该版本差异。
- Git 工作区干净；没有修改 EBGAD 算法代码。

## 环境

- Windows 独立虚拟环境：.venvs/ebgad。
- Python 3.11.16；PyTorch 2.4.1+cpu；PyG 2.6.1。
- NumPy 2.3.5；SciPy 1.16.0；scikit-learn 1.7.1；PyGOD 1.1.0；PyAMG 5.3.0。
- WSL Python 3.8.10 不满足 EBGAD 的 Python >=3.10 要求，未用于本次运行。
- 原 Windows 通用环境 PyTorch 2.3.1 与 NumPy 2.x 出现 ABI 错误，已恢复 NumPy 1.26.4。本次环境与 README 的 Linux 测试环境有差异，正式比较需报告。

## 数据

- 数据集：PyGOD 的 Disney，首次运行自动下载。
- 缓存：C:/Users/28641/.pygod/data/disney.pt，20,643 bytes。
- SHA-256：16DCDE35C29F228B7666C91487271F1B3FEE533ADED38571AFBE683237DAF68C。
- 运行形状：124 个节点、28 维特征、6 个异常节点；124 个标签节点参加评估。
- 无额外预处理。Disney 用于 README 快速自检，不是论文 11 个主基准之一。

## 命令与结果

PowerShell 命令，从 EBGAD 仓库根目录执行：

    $repo = "C:\Users\28641\Desktop\论文科研-江权学长\Graph-Anomaly-Reproduction\Repositories\EBGAD"
    $py = "C:\Users\28641\Desktop\论文科研-江权学长\Graph-Anomaly-Reproduction\.venvs\ebgad\Scripts\python.exe"
    Set-Location $repo
    $env:PYTHONPATH = $repo
    & $py run_ebgad.py --dataset disney --permute-check --null-check 30 --out results/ebgad

| 记录项 | 结果 |
|---|---|
| 标签选择分数 | R |
| AUROC | 47.88% |
| AP/AUPRC | 5.25% |
| 选择自由扫描 SCAN | AUROC 45.62%，AP/AUPRC 5.16% |
| 置换检查 | 最小候选 Spearman 1.0000；所选分数 1.0000 |
| 零假设检查 | B=30；中位 KS 0.0153；程序参考线 0.0267 |
| 总耗时 | 未单独计时；首次运行包含下载和缓存 |

代码用 scikit-learn average_precision_score 计算 AP/AUPRC，终端按百分数打印。结果保存于 Repositories/EBGAD/results/ebgad/disney.json 和 disney_scores.npz。自检通过只说明程序链路及这两种检查正常，不代表论文检测指标已经复现。

## 结论和待办

当前完成独立环境和一次端到端冒烟测试，**尚未复现论文主表**。

- [ ] 将论文主结果表各数据集映射到数据来源、预处理、脚本、参数和配置。
- [ ] 区分 neurips2026 下的最终实验脚本与标记为 prototype 的探索脚本。
- [ ] 选择一个规模适中的主表数据集，锁定输入版本和随机种子后运行。
- [ ] 按论文次数重复，比较均值、标准差和指标实现。
- [ ] 主表核对后再跑消融与基线。
