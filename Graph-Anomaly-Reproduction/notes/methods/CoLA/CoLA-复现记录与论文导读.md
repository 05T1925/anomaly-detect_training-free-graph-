# CoLA 复现记录与论文导读

## 复现状态

逐段代码讲解、重跑命令、多 seed 实验流程和论文阅读路线见：[CoLA 详细复现操作与代码逐段导读](CoLA-详细复现操作与代码逐段导读.md)。

已经用仓库默认配置，在 Cora 注入异常数据上完成一次完整的 CoLA 运行：100 个训练 epoch、256 轮测试子图采样，最终 `AUC:0.8946`。本次采用 seed=1。论文 Table III 报告 Cora AUROC `0.8779`，是 10 次运行的平均值；当前单次结果不能直接和论文均值视为完全复现，后续要用多个 seed 比较均值。

运行耗时约 4.5 分钟（CPU）。训练中保存的 checkpoint 已移到 `results/CoLA/cora-seed1-best_model.pkl`，避免放在源码仓库根目录。

源码仓库当前短 commit：`c2b5273`。

## 环境与 DGL 版本

CoLA 和 TFGAD、FreeGAD 的执行方式不同：它有训练过程，并依赖 DGL 的旧随机游走 API。代码 README 要求 DGL 0.4.0；较新的 DGL 1.x/2.x 没有 `dgl.contrib.sampling.random_walk_with_restart`，不能直接替代。

已在 WSL Ubuntu 20.04 建立独立旧版环境 `~/.venvs/cola`：

| 包 | 版本 |
|---|---|
| Python | 3.7.9 |
| PyTorch | 1.8.1+cpu |
| DGL | 0.4.0 |
| NumPy | 1.21.6 |
| SciPy | 1.7.3 |
| scikit-learn | 1.0.2 |
| NetworkX | 2.5.1 |
| tqdm | 4.59.0 |

这是按仓库 README 旧依赖配置的 CPU 环境。论文正文的实现环境另写为 PyTorch 1.4.0 与 DGL 0.3.1；这次优先按 README 的 DGL 0.4.0 跑通，环境差异要记录在复现说明中。

## 第一步：生成异常注入数据

CoLA 原仓库的 `dataset` 初始只有 `ACM.mat`。Cora 的原始引文网络 pickles 在 `raw_dataset/cora/`，需要先注入异常：

```bash
cd "/mnt/c/Users/28641/Desktop/论文科研-江权学长/Graph-Anomaly-Reproduction/Repositories/CoLA"
~/.venvs/cola/bin/python inject_anomaly.py --dataset cora
```

已运行成功，生成 `dataset/cora.mat`。默认设置使用随机 seed=1、每个结构异常团体 15 个节点、5 个结构团体、属性异常挑选 50 个候选节点；总计注入 150 个异常，其中 75 个结构异常、75 个属性异常。脚本输出的结构增量 1050 是对称邻接矩阵项数（无向边两次计数；组内原本没有边时相当于 525 条无向边）。生成文件是 2708 节点、1433 维特征，异常标签 2558 个正常 / 150 个异常。

首次生成后不要重复运行 `inject_anomaly.py` 覆盖文件，除非明确要用新 seed 生成另一份实验数据。否则数据变化会影响与当前结果的比较。

## 第二步：完整运行 Cora

CoLA 的相对数据路径是 `./dataset`，所以工作目录必须是仓库根目录：

```bash
cd "/mnt/c/Users/28641/Desktop/论文科研-江权学长/Graph-Anomaly-Reproduction/Repositories/CoLA"
~/.venvs/cola/bin/python run.py --dataset cora --seed 1
```

`run.py` 的 `--dataset cora` 会限制到 Cora 单个数据集。它使用的默认/论文参数为：学习率 `0.001`、100 epochs、batch size 300、embedding dim 64、subgraph size 4、negative sampling ratio 1、readout=`avg`、256 轮测试采样。运行完成会在仓库根目录写入 `best_model.pkl`。

开发时可把测试采样轮数暂时调低，例如 `--auc_test_rounds 8`，用来检查环境和数据；这只是快速冒烟检查，不应当把结果当作论文设置的完整复现。论文设置是 256 轮。

PowerShell 进入 WSL：

```powershell
wsl -d Ubuntu-20.04
```

在 Ubuntu 中可直接用 `~/.venvs/cola/bin/python`，不需要先激活环境。也可以用 Micromamba 激活：

```bash
micromamba activate ~/.venvs/cola
```

## 论文应该怎么看

本地论文：[CoLA-TNNLS-2022.pdf](../../../Papers/methods/CoLA-TNNLS-2022.pdf)。Sections IV-A 到 IV-C 是方法主体，Section V-A 是实验设置。Table II 给出数据统计，Table III 是主结果，Table IV-VI 是消融实验。全文文本提取副本在同目录的 `CoLA-paper-extracted.txt`，可以搜索定位。

按这个顺序读最容易把论文映射到代码：

1. **Introduction**：理解论文为什么认为重构式 GAD 有不足，以及为什么要用“目标节点—局部子图”的对比实例。
2. **Section IV-A: Contrastive Instance Pair Definition**：读随机游走子图、目标节点屏蔽和正负样本构造。
3. **Section IV-B: Contrastive Self-Supervised Learning Model**：读局部子图上的 GCN、readout、判别器和 BCE 目标。
4. **Section IV-C: Anomaly Score Computation**：读测试阶段如何多轮抽样、统计正负匹配分数并转换成异常分数。
5. **Section V-A、Tables II–III**：核数据构造、训练轮数、测试轮数和论文报告的 AUC。
6. **Tables IV–VI**：了解平均 readout、正负实例分数和多轮均值为何被选择。

### 代码阅读顺序

| 文件/函数 | 作用 |
|---|---|
| `inject_anomaly.py::load_citation_datadet` | 从 Cora 原始 pickles 组装图、特征和类别标签 |
| `inject_anomaly.py` 主流程 | 构造结构异常团体与属性异常节点，并保存 MAT 文件 |
| `utils.py::load_mat` | 读图、属性、类别标签及异常标签，并拆分索引 |
| `utils.py::generate_rwr_subgraph` | 用 DGL RWR 为每个目标节点采局部子图 |
| `model.py::GCN` / `AvgReadout` / `Discriminator` | 局部图编码、邻域汇总和正负实例判别 |
| `run.py` 训练循环 | 构造正负 pair、计算 BCE、反向传播、按损失保存最佳 epoch |
| `run.py` 测试循环 | 重复采样 256 次，统计目标节点与邻域子图的匹配分数，计算 AUROC |

### 用一句话抓住 CoLA

对每个目标节点取它周围的局部子图，训练 GNN 让“节点与自己的局部子图”匹配、与错配的局部上下文不匹配；测试时，若目标节点与邻域的匹配程度低，就更可能是异常。训练标签没有参与优化；类别标签也不参与异常分数，只在最终计算 AUROC 时使用异常标签。

和前面两个方法的区别：CoLA 是训练式方法，必须训练模型；TFGAD 和 FreeGAD 则没有训练 epoch 和梯度优化。

## 当前结果如何解释

- 当前 Cora 数据由官方注入脚本 seed=1 生成，与论文报告的 150 个异常数量一致。
- 当前完整运行使用 seed=1，得到 AUC 0.8946；论文报告 10 次运行的均值 0.8779。
- 论文算法测试有 256 轮随机采样；运行时多轮采样不是训练，它是对测试分数做 Monte Carlo 平均。
- 单次结果高于论文均值可能来自模型随机种子、初始化或数据注入版本；不要仅拿一次高低判断实现好坏。要复现论文均值，应运行多个 seed，并保持注入数据和参数一致。
- 本仓库的 `auc_test_rounds` 只用于测试多轮采样，不是 train epoch。

## 下一步建议

1. 先把 `run.py` 这次运行中的主要模块逐段读完，特别是正 pair、负 pair、BCE loss 和测试异常分数的符号方向。
2. 固定 `dataset/cora.mat` 不变，运行 seed 2、3 等不同模型种子，记录每次 AUC；不要重新生成数据。
3. 汇总多 seed 的均值和标准差，再与论文 Table III 的 0.8779 做比较。若准备严格比较论文的 10 次均值，跑满 10 个 seed；默认每次约 4.5 分钟。
4. 再尝试 Citeseer/Pubmed，注入参数由脚本根据数据集自动设定；BlogCatalog/Flickr 使用 `raw_dataset/<name>/<name>.mat` 的特征/网络，脚本支持注入。
5. 将 seed、注入配置、`run.py` 参数、AUC、运行时间和当前源码 commit 写进实验记录。
