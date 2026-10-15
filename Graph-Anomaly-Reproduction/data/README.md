# 共享数据集目录

这里集中保存各复现实验可复用的数据副本。按数据集名称分文件夹；同名但处理方式不同的版本保留不同文件名，不覆盖。机器可读清单见 [`manifest.csv`](manifest.csv)，文件完整性校验值见 [`checksums.sha256`](checksums.sha256)。前沿论文的数据覆盖、获取状态和跨论文对应关系见 [`论文—数据集对应表`](../notes/references/paper-dataset-crosswalk.md)。

## 前沿论文新增数据包（2026-10-12）

下列补充数据针对 EB-GAD、AG-Forge/TS-GGAD 等近期工作；文件已放入共享目录，可供后续实验复用。这里的“已准备”表示数据文件在本机，并不表示已经运行对应模型或复现论文结果。

| 数据 | 本地位置 | 数据内容与使用提示 |
|---|---|---|
| Weibo / PyGOD 发布版 | `Weibo/weibo-pygod.pt.zip`、`Weibo/raw-pygod/weibo.pt` | PyTorch Geometric `Data` 序列化文件，读取需要兼容的 PyTorch/PyG 环境。8,405 节点、407,963 边、400 维特征。PyGOD 数据说明记录 868 个异常；EB-GAD 论文报告其加载版本有 347 个异常。标签版本有冲突，实验前应从本地 `weibo.pt` 读取标签并把实际计数写入日志。 |
| Elliptic++ Transactions | `EllipticPlusPlus/Transactions/txs_{features,classes,edgelist}.csv` | 官方交易图，203,769 个交易节点、234,355 条边。原始特征表列数与“特征维数”统计口径不同；引用论文时注明是否把时间步、交易 ID 计入特征数。 |
| Elliptic 全量标准版（派生） | `Elliptic/elliptic-original-reconstructed/` | 使用 EB-GAD 仓库的 `tools/reconstruct_elliptic.py` 从上述 Elliptic++ CSV 重建：203,769 节点、234,355 边、166 维特征；类别 1/2/3 映射为 illicit/licit/unknown。属于派生文件，不是另一次独立下载。 |
| Amazon 多关系 CARE-GNN 版 | `Amazon/care-gnn/Amazon.zip`、`Amazon/care-gnn/extracted/Amazon.mat` | 作者仓库发布包，包含多种关系矩阵和 `homo`。EB-GAD 统计为 11,944 节点、4,398,392 边、25 维特征、821 个异常。与 10,224 节点的 TAM/FreeGAD Amazon 不是同一数据版本。 |
| YelpChi 多关系 CARE-GNN 版 | `YelpChi/care-gnn/YelpChi.zip`、`YelpChi/care-gnn/extracted/YelpChi.mat` | 作者仓库发布包，包含 `rur/rtr/rsr/homo` 关系矩阵。EB-GAD 统计为 45,954 节点、3,846,979 边、32 维特征。与 NeighborDiv/AG-Forge 所用的 23,831 节点 YelpChi 子版本不同。 |
| AlignGAD 随仓库发布的数据副本 | 各数据集下的 `*-aligngad.mat` | 从本地 AlignGAD 作者仓库复制，文件名保留来源标识。Amazon、Cora、CiteSeer、CS、Photo、PubMed、Reddit、YelpChi 等与其他来源的同名文件分开保存；严格复现时先核对其标签、边数和预处理。 |

尚未下载：**DGraphFin**。它是 EB-GAD 的大型金融基准（3,700,550 节点、约 4.3M 边、17 维特征），官方获取流程需要在 DGraph 网站完成免费注册。我们没有代替你注册或绕过访问控制；后续可按 [`DGraphFin/README.md`](DGraphFin/README.md) 的说明自行申请后放入共享目录。

## NeighborDiv 论文数据

论文 Appendix B.1 列出 4 个 source graph（供训练式对比方法使用）和 7 个 target graph。NeighborDiv 本身是 training-free：运行它不需要 Facebook、Amazon、PubMed、Elliptic 这 4 个 source graph；target 的标签只用于最后计算 AUROC/AP，不参与打分。[论文全文](https://arxiv.org/abs/2605.20879)

| 角色 | 数据集 | 论文统计（节点 / 边 / 特征） | 本目录文件与状态 |
|---|---|---:|---|
| Source | Facebook | 1,081 / 55,104 / 576 | `Facebook/facebook-tam.mat`，规模和标签数均匹配 |
| Source | Amazon | 10,244 / 175,608 / 25 | `Amazon/amazon-freegad.mat`：实际 10,224 节点、693 个异常；边/特征/标签与该版本匹配。论文表的 10,244 与本地公开版本存在 20 节点差异，需注明版本；`amazon-ggad.mat` 是另一个 11,944 节点的 Amazon-all 版本 |
| Source | PubMed | 19,717 / 44,338 / 500 | `PubMed/planetoid-raw/` 保存 Planetoid 原始数据；其中原始标签是论文主题类别，不是论文所述的 600 个注入异常标签 |
| Source | Elliptic | 46,564 / 73,248 / 93 | `Elliptic/elliptic-ggad.mat`，规模、特征维度和 4,545 个异常与论文表匹配 |
| Target | Cora | 2,708 / 5,429 / 1,433 | `Cora/cora-cola.mat`、`Cora/cora-freegad.mat` 均有 2,708 节点、1,433 维特征、150 个异常，但分别有 5,803/5,802 条无向边，与论文表中的 5,429 不同；`Cora/inj_cora-tfgad.bin` 是 TFGAD 的另一种注入版本 |
| Target | Reddit | 10,984 / 168,016 / 64 | `Reddit/reddit-freegad.mat` 的规模和 366 个异常与论文表匹配；另存 `Reddit/inj_reddit-tfgad.bin` 供 TFGAD 复用 |
| Target | YelpChi | 23,831 / 49,315 / 32 | `YelpChi/yelpchi-freegad.mat`，节点、边、特征和 1,217 个异常与论文表匹配。MAT 邻接矩阵有 98,630 个有向存储项，即 49,315 条无向边 |
| Target | T-Finance | 39,357 / 21,222,543 / 10 | `T-Finance/tfinance-source.zip` 保留原始压缩包；共享目录也已有 `T-Finance/tfinance-source/tfinance` 解压文件。论文表报告 1,803 个异常 |
| Target | Tolokers | 11,758 / 519,000 / 10 | `Tolokers/tolokers.npz`，原始图、特征、标签和 10 组 split 均已保存；标签中有 2,566 个正类 |
| Target | Disney | 124 / 335 / 28 | `Disney/disney-dominant.mat`，含 6 个异常，匹配论文表 |
| Target | Questions | 48,921 / 153,540 / 301 | `Questions/questions.npz`，原始图、特征、标签和 10 组 split 均已保存；含 1,460 个正类 |

Tolokers/Questions 的 NPZ 来自 [Yandex heterophilous-graphs 官方数据](https://github.com/yandex-research/heterophilous-graphs/tree/main/data)，也是 PyTorch Geometric 的 `HeterophilousGraphDataset` 使用的发布版本；NPZ 中保留的 split 是节点分类基准 split，不等同于 NeighborDiv 的异常检测训练划分。[PyG 数据加载说明](https://pytorch-geometric.readthedocs.io/en/2.5.2/_modules/torch_geometric/datasets/heterophilous_graph_dataset.html)

Yandex 官方加载器会补成双向图。但 2026-10-13 实测发现：NeighborDiv Table 16 的 Tolokers/Questions 度数统计逐项匹配原始 NPZ 的入度，双向图则不匹配。NeighborDiv 通用加载器 `auto` 仍按官方方式处理，论文实验配置使用显式 `incoming` 作为候选协议；这项推断尚不等于作者实际输入已确认。Disney 的无向并集已使主指标对齐；Cora 的 FreeGAD/AlignGAD 副本匹配 Table 16，CoLA 副本不匹配。见 [根因复检](../notes/methods/NeighborDiv/NeighborDiv-指标差距根因复检-2026-10-13.md)。

T-Finance 压缩包来自 [BWGNN/ICML 2022 作者仓库](https://github.com/squareRoot3/Rethinking-Anomaly-Detection)所链接的[公开数据目录](https://drive.google.com/drive/folders/1PpNwvZx_YRSCDiHaBUmRIS3x1rZR7fMr?usp=sharing)。Elliptic 处理版从 [GGAD 作者仓库](https://github.com/mala-lab/GGAD)链接的[公开数据目录](https://drive.google.com/drive/folders/1rEKW5JLdB1VGwyJefAD8ppXYDAXc5FFj?usp=sharing)取得。Facebook 数据与 Amazon 公开副本来自 [TAM 作者仓库](https://github.com/mala-lab/TAM-master)的[公开数据目录](https://drive.google.com/drive/folders/1qcDBcVdcfAr_q5VOXBYagtnhA_r3Mm3Z?usp=drive_link)。新增 Weibo 包来自 [PyGOD 官方数据仓库](https://github.com/pygod-team/data)，Elliptic++ CSV 来自[官方代码仓库](https://github.com/git-disl/EllipticPlusPlus)提供的[官方数据目录](https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l?usp=sharing)，Amazon/YelpChi 多关系包来自 [CARE-GNN 作者仓库](https://github.com/YingtongDou/CARE-GNN)。使用或分享数据时，请遵守各原始数据源的引用和再分发要求。

## 其他复现实验的共享数据

此前 TFGAD、CoLA、FreeGAD、DOMINANT 已用到的版本也复制到这里，便于后续统一定位：

- ACM、BlogCatalog、Enron：DOMINANT/CoLA 的 MAT 版本。
- Flickr：同时保存 DOMINANT 与 FreeGAD 版本；两者标签数和邻接项数有差异，不要混用。
- Amazon、Cora、Reddit：保存此前仓库的 MAT 版本；Cora、Reddit 还保留 TFGAD 的 DGL `.bin` 注入版本。

原始仓库中的数据文件未移动或删除；这些是供工作区共用的副本。复现实验时应在记录中写清楚实际使用的文件路径、SHA-256、图预处理和标签协议。同名不代表同一版本，尤其是 Cora、Amazon、YelpChi、Flickr。

## T-Finance 解压

需要读取 DGL 图时，在 PowerShell 中解压到单独的目录（会额外占用约 683 MB）：

```powershell
Expand-Archive -LiteralPath .\data\T-Finance\tfinance-source.zip -DestinationPath .\data\T-Finance\raw
```

解压后文件名为 `tfinance`（没有扩展名），是 DGL 图文件。NeighborDiv 支持 `--format dgl` 直接读取，或 `--format dgl-zip` 临时解压。加载器已经用于完整实验并核对：39,357 节点、42,445,086 双向邻接项、39,275 有效节点、1,804 异常标签。论文标签数为 1,803，但单标签差异不足以解释 AP 差距，详见根因复检。只有使用这些格式时才需要兼容的 torch/DGL 环境；必须保留原始节点顺序。
