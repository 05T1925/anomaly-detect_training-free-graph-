# CoLA 详细复现操作与代码逐段导读

这份指南接在 [CoLA 复现记录与论文导读](CoLA-复现记录与论文导读.md) 后面，面向已经跑通一次 Cora 的复现者。重点是把数据、论文中的方法、仓库代码和实验结果连起来。

## 1. 先用一句话理解 CoLA

CoLA 对每个节点 `v` 建立两种配对：节点本身与它附近的局部子图是正样本；节点本身与别处采来的局部上下文是负样本。模型学习区分这两类配对。正常节点通常和自己的邻居上下文匹配得较好；属性或连接关系异常的节点，其正负配对分数往往更难拉开。测试时把“负配对分数减正配对分数”作为异常程度，再对多轮采样取平均。

这是一种**无监督的自监督训练**：训练标签是程序自动构造的正/负配对标签，不是人工异常标签。人工异常标签只在最后计算 AUC 时用于评估。

## 2. 在现有 Cora 数据上复跑

当前工作目录是 Windows 桌面路径。先从 PowerShell 进入已有的 WSL 环境：

```powershell
wsl -d Ubuntu-20.04
```

在 Ubuntu 中执行完整默认实验：

```bash
cd "/mnt/c/Users/28641/Desktop/论文科研-江权学长/Graph-Anomaly-Reproduction/Repositories/CoLA"
~/.venvs/cola/bin/python run.py --dataset cora --seed 1
```

这会读取现有 `dataset/cora.mat`，训练 100 个 epoch，再做 256 轮测试采样。它会把当前最佳 checkpoint 写到仓库根目录的 `best_model.pkl`，因此再次运行会覆盖这个临时 checkpoint。已有的 seed=1 checkpoint 已另存到 `results/CoLA/cora-seed1-best_model.pkl`。

可先检查旧环境是否仍可用：

```bash
~/.venvs/cola/bin/python -c "import torch,dgl; print('torch',torch.__version__,'dgl',dgl.__version__)"
```

项目使用旧版 DGL 的 `dgl.contrib.sampling.random_walk_with_restart`。目前记录的可运行组合是 Python 3.7.9、PyTorch 1.8.1 CPU、DGL 0.4.0。不要把这个环境升级到 DGL 1.x/2.x 后直接运行；新版本移除了该旧接口。论文实验环境写的是 PyTorch 1.4.0 / DGL 0.3.1，仓库 README 要求 DGL 0.4.0，所以运行环境差异要写进复现记录。

### 生成 Cora 注入数据

只有需要重建数据时才运行下面这一步：

```bash
cd "/mnt/c/Users/28641/Desktop/论文科研-江权学长/Graph-Anomaly-Reproduction/Repositories/CoLA"
~/.venvs/cola/bin/python inject_anomaly.py --dataset cora --seed 1
```

它会覆盖 `dataset/cora.mat`。本次复现实验已经有该文件，所以普通重跑只执行 `run.py`。生成 seed 和训练 seed 是两件事：`inject_anomaly.py --seed` 决定异常节点/扰动数据；`run.py --seed` 决定训练与采样随机过程。想比较模型随机性时，固定现有 MAT 文件，只改变 `run.py --seed`。

当前 Cora 文件包含 2708 个节点、1433 维特征和 150 个异常节点：75 个结构异常、75 个属性异常。默认 `m=15, n=5, k=50`：抽取 5 组、每组 15 个节点，把每组内部变成全连接小团体；再抽取 75 个节点，从 50 个候选节点中选属性欧氏距离最远者，把它的特征拷给异常节点。脚本打印的“增加 1050”是对称邻接矩阵的非零项总数增量；在每组原本没有组内边时，对应 525 条无向边，因为无向边在矩阵里计两次。

## 3. 默认实验参数和怎么做多次运行

| 参数 | Cora 默认值 | 含义 |
|---|---:|---|
| `lr` | 0.001 | Adam 学习率 |
| `num_epoch` | 100 | 训练轮数 |
| `batch_size` | 300 | 每个训练 batch 的目标节点数 |
| `embedding_dim` | 64 | 节点/子图嵌入维度 |
| `subgraph_size` | 4 | RWR 子图节点数；含被匿名化的锚点 |
| `negsamp_ratio` | 1 | 每个正配对对应的负配对数 |
| `readout` | `avg` | 局部子图的平均池化 |
| `auc_test_rounds` | 256 | 测试采样轮数，不是训练轮数 |

`--auc_test_rounds 8` 可以缩短调试时的测试阶段，但仍会训练 100 个 epoch；这样的 AUC 不应拿来和论文主结果比较。论文报告 Cora AUC 0.8779，是 10 次运行的平均值。本次 seed=1 的单次结果是 0.8946，不能据此判断复现优劣。

若要得到固定这份 `cora.mat` 上的 10 个训练随机种子结果，可在 WSL 中串行执行，并逐个保存日志和 checkpoint：

```bash
set -euo pipefail
cola_root="/mnt/c/Users/28641/Desktop/论文科研-江权学长/Graph-Anomaly-Reproduction"
cd "$cola_root/Repositories/CoLA"
mkdir -p "$cola_root/results/CoLA"

for seed in {1..10}; do
  ~/.venvs/cola/bin/python run.py --dataset cora --seed "$seed" 2>&1 \
    | tee "$cola_root/results/CoLA/cora-seed${seed}.log"
  cp best_model.pkl "$cola_root/results/CoLA/cora-seed${seed}-best_model.pkl"
done
```

这套方案固定异常注入数据，只变训练随机种子。论文只说运行 10 次并报告平均值，没有给出可复用的 seed 列表；因此要把这个选择写进实验记录。每次约 4.5 分钟，10 次约 45 分钟。日志中搜索 `AUC:` 可抄录每次结果，再报告均值和标准差。

## 4. 逐段看代码：从原始数据到 AUC

### A. `inject_anomaly.py`：把干净图变成带已知真值的测试数据

1. `load_citation_datadet` 读取 Cora 的 Planetoid pickle 文件，重排测试节点，组装特征矩阵、邻接矩阵和原始类别标签。
2. 主流程按 seed 随机抽异常节点，分别标记 `str_anomaly_label` 和 `attr_anomaly_label`，合并成 `Label`。
3. 结构扰动把每组 15 个结构异常节点内部连成团；属性扰动把异常节点特征替换成远处候选节点的特征。
4. `scipy.io.savemat` 保存 `Network`、`Attributes`、原类别 `Class` 和异常标签到 `dataset/cora.mat`。

这里模拟的是**合成异常**。结构异常具体体现为异常节点彼此形成稠密小团；属性异常具体体现为特征被远处候选节点的特征替换。它们是论文评测协议，不等同于真实应用里所有异常的定义。

### B. `utils.py`：读取、归一化及 RWR 采样

- `load_mat` 读取 MAT 文件，返回图、特征、原类别标签、异常标签和随机生成的索引划分。注意：`run.py` 后续并没有用 `idx_train / idx_val / idx_test` 做监督训练或留出测试；训练目标是自监督配对分类，且遍历全部节点。
- `preprocess_features` 做特征行归一化。
- `normalize_adj` 对邻接矩阵做 `D^(-1/2) A D^(-1/2)` 对称归一化；`run.py` 随后再加自环。RWR 子图从原图采样，GCN 使用归一化并加自环后的邻接矩阵。
- `generate_rwr_subgraph` 用 DGL RWR 为每个节点抽局部节点，去重后取固定数量；目标节点最后追加。若局部唯一节点不够，会重试或重复已有节点以满足定长输入。

### C. `run.py`：构造正负样本并训练

每轮训练先为每个节点采一个正子图，然后将所有节点打乱并按 batch 取出。`subgraph_size=4` 时，采样出的子图由 3 个局部节点加目标节点组成。代码把子图内目标节点的特征清零，避免模型只靠“目标节点是否出现在子图”投机；同时在子图外增加一个保留真实特征的目标节点表示。于是送进 GCN 的输入有 5 行：3 个邻居、1 个被匿名化的子图锚点、1 个独立目标节点。

`model.py` 的 `GCN` 用线性映射、邻接聚合、偏置和 PReLU 计算节点嵌入。`AvgReadout` 对前 4 行求平均，形成局部子图表示；第 5 行是目标节点表示。`Discriminator` 用双线性打分函数判断二者是否匹配。

正样本分数来自每个目标节点自己的局部子图。仓库实现里的负样本由 `Discriminator` 将 batch 中的子图上下文循环错位后得到：节点仍是当前 batch 中的目标，但上下文来自另一个目标的子图。也就是说，负配对不是用异常标签挑选出来的，而是随机错配的自监督例子。

每个 batch 的标签是“正配对为 1、错配负配对为 0”，用 `BCEWithLogitsLoss` 计算二元交叉熵，再由 Adam 更新参数。注意输入 loss 的是未过 sigmoid 的 logits；测试时才对分数取 sigmoid。程序每轮记录训练损失最低的 checkpoint。`cnt_wait` 虽然统计损失没有改善的轮数，但没有触发提前停止；`idx_val` 也没有参与模型选择。

### D. 测试：把匹配差变成异常分数

测试阶段加载最低训练损失对应的模型。每轮重新采子图，并以 `sigmoid` 将正、负配对 logits 转成分数。`run.py` 中的

```python
ano_score = - (logits[:cur_batch_size] - logits[cur_batch_size:])
```

等价于论文的 `s_negative - s_positive`。分数越大，表示模型越难把该节点和自身局部上下文区分于错配上下文，异常程度越高。程序对 256 轮的分数逐节点求均值，最后用 `roc_auc_score(ano_label, ano_score_final)` 计算 AUC。

AUC 是排序指标：可以理解为随机抽一个异常节点和一个正常节点，模型把异常节点排在前面的概率。AUC=0.5 约等于随机排序，越接近 1 越好；它不需要先选一个“异常/正常”的阈值。`Class` 类别标签在本次训练目标和最终 AUC 中都没有使用；最终评估使用的是 `Label` 异常标签。

## 5. 论文怎么读，以及对应哪些公式/表

建议带着问题读，而不是从头逐字啃：

1. **Introduction**：为什么重构式图自编码器可能不是最直接的异常目标？CoLA 选择让模型判别什么？重点看“节点—局部邻域不一致”这一动机。
2. **Section III**：熟悉 attributed network 的记号：图结构 `A`、属性矩阵 `X`、节点 `v_i`。
3. **Section IV-A（公式 1–3，Figure 3）**：什么叫 `target node vs. local subgraph`？正/负 pair 如何采样？为什么对子图中的锚点做匿名化？
4. **Section IV-B（公式 4–10，Figure 2）**：GCN 得到局部节点嵌入；readout 汇总局部子图；双线性 discriminator 输出匹配分数；BCE 训练正负 pair。
5. **Section IV-C（公式 11–12）**：单次分数是 `s_negative - s_positive`；多轮抽样后取均值，降低一次局部采样不完整带来的波动。
6. **Section IV-D–E**：算法流程和复杂度分析。论文强调小批量 pair 避免把整图送入 GCN。
7. **Section V-A–C（Table II–III）**：Table II 看基准数据与合成异常规模；Table III 看主结果。论文报告 Cora 的 CoLA AUC 为 0.8779，10 次运行取均值。
8. **Section V-D–E（Tables IV–VI）**：读参数研究和消融，理解为什么默认子图大小为 4、readout 用平均池化、测试轮数用 256，以及只用正分或负分各有什么损失。

### 论文和仓库实现的复现边界

- 论文宣称小批量 pair 的 GNN 计算空间不必随完整图节点数增长；但本仓库 `run.py` 把归一化邻接矩阵执行 `todense()` 后放进稠密 Tensor。因此局部 GCN 的 batch 激活是小的，数据加载阶段仍会保留 `N × N` 稠密邻接矩阵。复现大图时，不要把论文的模型复杂度表述直接当成这份代码整体内存占用的保证。
- 论文写明运行 10 次报告平均值；本地 seed=1 单次 AUC=0.8946，Table III 是 0.8779。首先固定数据并多跑训练 seed，报告均值/标准差；不要以单次偏高或偏低判断实现是否正确。
- 论文报告的运行环境与仓库 README 不完全相同，且本复现使用 CPU。把这些差异和源码 commit `c2b5273` 一起记在实验记录中。

## 6. 一张纸的复现记录模板

每次结果至少记录：

```text
代码 commit：c2b5273
数据集：Cora / dataset 注入 seed=1（固定 cora.mat）
模型 seed：1
环境：Python 3.7.9 / PyTorch 1.8.1+cpu / DGL 0.4.0
参数：lr=0.001, epochs=100, batch=300, dim=64, subgraph=4, rounds=256
AUC：0.8946
运行时间：约 4.5 分钟（CPU）
备注：训练遍历全部节点；测试标签只用于最终 AUC
```

当以上术语都能不看代码解释清楚时，CoLA 的主流程就基本掌握了：**生成带真值的评测数据 → 用“配对是否匹配”做自监督训练 → 用匹配差的多轮均值排序节点 → 用隐藏异常标签计算 AUC。**
