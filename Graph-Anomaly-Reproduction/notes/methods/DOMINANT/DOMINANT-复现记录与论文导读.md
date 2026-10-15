# DOMINANT 复现记录与论文导读

## 当前复现状态

已克隆作者提供的 PyTorch 仓库，并用已有的 CoLA Python 3.7.9 环境在 BlogCatalog 数据上完成 100 epoch 实验。代码使用 CPU，参数为 `hidden_dim=64, lr=0.005, dropout=0.3, alpha=0.8`。完整运行日志保存在 `results/DOMINANT/blogcatalog-seed-unfixed.log`；最终 AUC 为 0.8142。本次运行没有固定随机种子，因此结果仅是代码跑通的单次基线，不应拿来声称严格复现论文。

一次 epoch 的快速检查 AUC 为 0.8105。100 epoch 运行中第 30、40、50、60、90 轮 AUC 分别约为 0.8141、0.8142、0.8142、0.8143、0.8143；最终 99 轮 AUC 为 0.8142。实验采用官方代码提供的 BlogCatalog 数据，数据矩阵尺寸为 5,196 个节点、8,189 个属性；代码默认会在输入邻接矩阵上加自环并作对称归一化。

## 代码和论文版本的区别

作者最初的 TensorFlow 仓库要求 Python 2.7.3 和 TensorFlow 1.x，环境很旧；README 指向作者后续的 PyTorch 仓库。本次使用后者，短 commit 为 `83fa939`。它在当前 CoLA 环境中能运行（Python 3.7.9、PyTorch 1.8.1 CPU、SciPy 1.7.3、scikit-learn 1.0.2），不需要单独安装 PyG 或 DGL。

需要明确：该 PyTorch 代码是可运行的作者版本，但仓库配置并不等同论文正文报告的全部配置。论文实验写的是 300 epochs，学习率 0.005，编码器为 64、32、16 三层；当前仓库的 `model.py` 编码器是两层且 `run.py` 默认训练 100 epochs。当前结果应表述为“作者 PyTorch 仓库 BlogCatalog 基线复现”，后续若要对齐论文，需要将训练轮数改为 300，并核实/实现论文中的层宽和数据处理，再固定多个随机种子报告均值与方差。

## 数据与复现命令

当前使用的数据来自作者 PyTorch 仓库自带的 `data/BlogCatalog.mat`，MAT 文件键名为 `Network`、`Attributes`、`Label`，无需重新下载或转换。BlogCatalog 是建议先跑的集合：作者代码能直接读，且相较 ACM 体量更适合 CPU。Flickr 也随仓库提供；ACM 有 16,484 个节点，代码将邻接矩阵和重构矩阵转成稠密张量，CPU 内存和耗时都会显著增加，不建议第一轮就跑 ACM。

在 WSL Ubuntu 中运行：

```bash
cd "/mnt/c/Users/28641/Desktop/论文科研-江权学长/Graph-Anomaly-Reproduction/Repositories/DOMINANT-PyTorch"
~/.venvs/cola/bin/python run.py --dataset BlogCatalog --device cpu --epoch 100 --lr 0.005 --alpha 0.8 --hidden_dim 64 --dropout 0.3
```

切换数据集时将 `BlogCatalog` 换成 `Flickr` 或 `ACM`。此脚本逐 epoch 打印训练损失，每 10 轮评估一次 AUC。当前仓库没有保存模型或结果文件的参数；需要保留结果时可这样重定向日志：

```bash
~/.venvs/cola/bin/python -u run.py --dataset BlogCatalog --device cpu --epoch 100 --lr 0.005 --alpha 0.8 --hidden_dim 64 --dropout 0.3 > ../../results/DOMINANT/blogcatalog-seed1.log 2>&1
```

注意：仓库没有设置 Python、NumPy、PyTorch 的随机种子，上述文件名中的 `seed1` 只有在你另外固定所有随机数之后才准确；可以先用 `seed-unfixed` 命名，避免实验记录错误。论文报告 300 epoch 的配置可以先改命令为 `--epoch 300`，但仍未解决网络层数不一致的问题。

## 模型要点

DOMINANT 面向带属性图中的节点异常。输入为邻接矩阵 `A` 和节点属性矩阵 `X`。一个共享 GCN 编码器从两者中学习节点表示 `Z`，两个解码器分别重构属性和结构：

- 属性解码器输出 `X_hat`，每个节点属性重构误差反映属性异常程度。
- 结构解码器通过内积 `Z Z^T` 输出重构邻接关系 `A_hat`，每个节点的行重构误差反映结构异常程度。
- 节点分数由属性与结构重构误差加权，`alpha` 越大越偏重属性，越小越偏重结构。代码中 `alpha=0.8`。
- `run.py` 使用逐节点的欧氏重构误差（平方误差求和后开平方）计算 score，并用带异常标签的注入数据算 ROC-AUC；标签只用于评估，不参与优化。

代码阅读顺序建议：

1. `run.py`：数据读取、模型创建、训练循环、AUC 计算。
2. `utils.py`：MAT 文件键名、邻接自环与对称归一化。
3. `model.py`：共享编码器、属性解码器、结构解码器以及前向传播。
4. `layers.py`：单层 GCN 的矩阵运算 `A_norm X W + b`。
5. 回到论文第 3 节：把编码、双重重构、异常打分与代码逐一对应。

读代码时可以在 `A_hat, X_hat = model(attrs, adj)`、`loss_func(...)`、`score = loss.detach()` 三处打断点，逐个查看矩阵维度：`attrs: (N,d)`、`adj: (N,N)`、`X_hat: (N,d)`、`A_hat: (N,N)`、`score: (N,)`。特别注意 `utils.py` 返回的 `adj` 已含自环，而模型传播用的是加自环后归一化的邻接；loss 对照 `adj_label`，也就是带自环的结构矩阵。

## 论文阅读与写作训练

正式论文为 Kaize Ding, Jundong Li, Rohit Bhanushali, Huan Liu, “Deep Anomaly Detection on Attributed Networks,” SDM 2019, pp. 594–602, DOI: `10.1137/1.9781611975673.67`。建议依次读：

1. 摘要和引言：问题是什么，传统子空间/ego-network 方法面对稀疏、非线性和属性—结构交互时有什么局限。
2. 第 2 节：明确图 `G=(V,E,X)`、邻接矩阵 `A`、特征矩阵 `X`，以及异常检测是对节点排序。
3. 第 3 节：重点理解 GCN encoder、结构解码 `sigmoid(ZZ^T)`、属性解码和加权异常分数。
4. 第 4 节：记录数据集、异常注入方式、对比方法、AUC 与 Precision/Recall@K。不要把代码输出的一次 AUC 与论文曲线/均值直接等同。
5. 第 5 节及参考文献：将 DOMINANT 与 Radar、ANOMALOUS、CoLA 等方法的“异常证据”和训练方式区分开。

练习写作时先完成四句话：研究缺口（旧方法难以联合表达结构和属性的非线性交互）；方法概述（共享 GCN 编码 + 属性/结构双重重构）；异常定义（节点越难被正常模式重构，分数越高）；实验结论（需依论文多数据集、多指标结果支撑，不能只凭单次跑通）。再扩写成一段摘要和一张“方法—代码—证据”表。

## 全文可获得性说明

可直接读的官方资源包括作者代码仓库、README 和作者主页。作者旧 ASU PDF 地址目前返回 HTML 错误页；作者主页的论文 PDF 指向 SIAM 出版页面，当前页面提示需要访问权限。项目里曾误放一份 UIUC 2024 硕士论文，其内容不是 DOMINANT 原论文，已改名为 [`NOT-DOMINANT-UIUC-Active-Graph-Anomaly-Thesis-2024.pdf`](../../../Papers/related-work/NOT-DOMINANT-UIUC-Active-Graph-Anomaly-Thesis-2024.pdf) 以免误用。暂时没有核验到可合法公开下载的 DOMINANT 原论文 PDF。不要把这篇硕士论文或网页预览当成原论文。可尝试学校图书馆的 SIAM 访问、馆际互借，或礼貌联系作者询问可分享的作者稿。

## 参考链接

- 作者 PyTorch 仓库：<https://github.com/kaize0409/GCN_AnomalyDetection_pytorch>
- 作者原始 TensorFlow 仓库：<https://github.com/kaize0409/GCN_AnomalyDetection>
- SIAM DOI / 出版页面：<https://doi.org/10.1137/1.9781611975673.67>
- 作者论文列表：<https://kaize0409.github.io/Publications.html>
- 论文官方作者主页列出的 PDF 链接：<https://epubs.siam.org/doi/epdf/10.1137/1.9781611975673.67>
