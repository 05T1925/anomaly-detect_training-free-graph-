# EBGAD 首次复现与论文代码学习

## 目前进度

EB-GAD（Empirical-Bayes Graph Anomaly Detection）是经验贝叶斯图异常检测方法。PyGOD 在这个仓库里主要帮助加载部分数据集；它不是 EB-GAD 算法本身。EB-GAD 不训练神经网络，但会拟合统计参数并计算节点异常分数。

截至 2026-10-12，已在独立 Windows 环境成功运行作者 README 的 Disney 小图 quick start，数据加载、拟合、打分、评估、置换检查和零假设模拟检查均已完成。这属于**冒烟测试**，还没有复现论文主表。

本地论文为 arXiv:2609.38424 v1，标题是 *Graph Anomaly Detection as Finite-Horizon Control: Training-Free Scoring via Empirical Bayes*。本地代码 commit 为 cba2969e390ae6cd4381fe86393fa6cfcfc40d13。仓库 README 仍写旧标题 *Graph Anomaly Detection as Dynamical Transport*，所以后续记录要同时注明论文版本和代码 commit。来源：[arXiv v1](https://arxiv.org/abs/2609.38424)、[作者代码仓库](https://github.com/heraclixus/EBGAD)。

## 环境与数据

WSL 当前 Python 3.8.10 低于仓库要求的 Python 3.10；原 Windows 通用环境的 PyTorch 2.3.1 与 NumPy 2.x 出现 ABI 兼容问题。因此另建了项目外层的 .venvs/ebgad，没有改动原环境。

| 软件 | 本次版本 |
|---|---|
| Python | 3.11.16 |
| PyTorch | 2.4.1+cpu |
| PyTorch Geometric | 2.6.1 |
| NumPy | 2.3.5 |
| SciPy / scikit-learn | 1.16.0 / 1.7.1 |
| PyGOD / PyAMG | 1.1.0 / 5.3.0 |

仓库 README 写的测试环境是 Linux CPU、Python 3.12、PyTorch 2.3.1 等；本机环境略有差异，正式对照论文时必须如实记录。PyTorch 的 Windows NumPy 2 兼容问题在 2.4.1 修复，见 [PyTorch issue #131668](https://github.com/pytorch/pytorch/issues/131668)。

Disney 数据由 PyGOD 首次运行时下载。本次图有 124 个节点、28 维特征、6 个异常节点。缓存文件 C:/Users/28641/.pygod/data/disney.pt 的 SHA-256 为 16DCDE35C29F228B7666C91487271F1B3FEE533ADED38571AFBE683237DAF68C。

## 运行命令

在 PowerShell 从 EBGAD 仓库根目录运行：

    $repo = "C:\Users\28641\Desktop\论文科研-江权学长\Graph-Anomaly-Reproduction\Repositories\EBGAD"
    $py = "C:\Users\28641\Desktop\论文科研-江权学长\Graph-Anomaly-Reproduction\.venvs\ebgad\Scripts\python.exe"
    Set-Location $repo
    $env:PYTHONPATH = $repo
    & $py run_ebgad.py --dataset disney --permute-check --null-check 30 --out results/ebgad

- --dataset disney：读取 PyGOD 的 Disney 小图。
- --permute-check：打乱节点编号后重跑，检查结果排序是否依赖编号。
- --null-check 30：用拟合出的模型模拟 30 组数据，检查理论零分布。
- --out results/ebgad：保存汇总 JSON 和每个节点的分数 NPZ。

## 运行结果

| 项目 | 输出 |
|---|---:|
| 标签选择分数 | R |
| 该分数 AUROC | 47.88% |
| 该分数 AP/AUPRC | 5.25% |
| 选择自由扫描 SCAN | AUROC 45.62%，AP/AUPRC 5.16% |
| 置换检查 | 候选最小 Spearman 1.0000；所选分数 1.0000 |
| 零假设检查 | B=30，中位 KS 0.0153；程序显示参考线 0.0267 |

代码用 scikit-learn 的 average_precision_score 计算 AP/AUPRC；终端按百分数打印，47.88 对应比例 0.4788。置换和零假设检查验证程序性质，不证明检测效果达到论文水平。Disney 是 quick start 自检图，不是论文主表的复现结果。输出文件在 Repositories/EBGAD/results/ebgad/disney.json 和 disney_scores.npz。Git 工作区干净，没有修改算法源码。

## 用直白的话理解方法

把图想成“谁和周围节点的常见模式不合”。EB-GAD 先从图结构和节点特征估计正常模式；再把节点特征想象成沿图的谱方向向平滑模板逐渐松弛。一个节点偏离模板越难用这套模型解释，它的控制能量分数就可能越异常。方法生成一组候选分数，再用不看标签的规则挑选；标签只在最后算指标时使用。“Training-free”是没有神经网络训练轮次、反向传播和梯度优化，不代表没有统计拟合和数值计算。

## 论文和代码阅读路线

| 顺序 | 论文 / 代码 | 初学时要弄清的问题 |
|---:|---|---|
| 1 | 论文 Abstract、Introduction | 图异常检测要找什么；作者指出现有方法哪里不稳或难解释？ |
| 2 | 论文 §3 方法 | 正常模板、经验贝叶斯拟合、有限时域控制能量、候选分数选择分别是什么？ |
| 3 | data_utils.py 的 load_data | 数据来自 PyGOD、MAT 还是欺诈 CSV？标签里是否有未标注节点？ |
| 4 | run_ebgad.py 的 run_once | 数据如何依次经过预处理、拟合、打分、无标签选择和评估？ |
| 5 | ebgad/prep.py 的 prepare | 图谱、模板、残差如何形成？谱分解为什么会花时间和内存？ |
| 6 | ebgad/fit.py 的 fit_prepared、select_gamma | 如何拟合统计参数并选择图尺度？ |
| 7 | ebgad/scores.py 的 compute_members、compute_bank | 节点如何得到多种候选异常分数？ |
| 8 | ebgad/nulls.py、ebgad/select.py 的 select | 理论零分布如何计算？怎样在不看标签时选择分数？ |
| 9 | run_ebgad.py 指标函数、eval_utils.py | 最终使用了哪些标签、哪些指标？ |
| 10 | 论文 §5、主结果表、消融表、附录 | 数据、参数、随机种子、重复次数和代码入口是否一一对应？ |

代码入口：[数据加载](../../../Repositories/EBGAD/data_utils.py:419)、[总流程](../../../Repositories/EBGAD/run_ebgad.py:108)、[预处理](../../../Repositories/EBGAD/ebgad/prep.py:266)、[经验贝叶斯拟合](../../../Repositories/EBGAD/ebgad/fit.py:91)、[分数计算](../../../Repositories/EBGAD/ebgad/scores.py:151)、[分数选择](../../../Repositories/EBGAD/ebgad/select.py:225)。

## 从自检走向论文复现

1. 对照论文 §5 和附录，确认主结果表每一行的数据源、预处理、参数、随机种子、运行次数和指标算法。
2. 在 DATA.md 与 data_utils.py 中查明数据读取路径，记录数据版本、节点数、边数、特征数、标签数和文件哈希。
3. 在 neurips2026/README.md 中找出对应表格的最终脚本与配置，先跑一个规模适中的主表数据集，再按论文次数重复并计算均值和标准差。
4. 核对结果后再扩大到其他数据集、消融和基线。

仓库 README 明确提醒：run_ebgad.py 默认值是 package defaults，不是论文逐数据集配置。论文表格由 neurips2026/ 下的脚本产生；README 推荐的 test_dynamic_score_bank.py 自称 prototype，而 final_paper_results.py 需要 sweep 生成的配置文件。因此不能把 Disney 自检或任意默认命令称为论文主结果复现。下一步要先把论文表格与具体脚本、配置逐项对应。
