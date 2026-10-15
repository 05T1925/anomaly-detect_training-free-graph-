# 前沿图异常检测论文—数据集—本地文件对应

更新时间：2026-10-12。本文用于规划后续复现，文件“已就绪”只表示本地数据已存放，不代表已运行论文模型。

## 论文覆盖

| 论文 / 方法 | 论文使用的数据集 | 本仓库覆盖情况 |
|---|---|---|
| [NeighborDiv（2026）](https://arxiv.org/abs/2605.20879)，training-free zero-shot GGAD | Source：Facebook、Amazon、PubMed、Elliptic。Target：Cora、YelpChi、Reddit、T-Finance、Tolokers、Disney、Questions。NeighborDiv 自身推理不需要 source graph，标签仅用于离线评价。 | 全部有本地文件。各自对应的多种来源、处理版本见 [`data/README.md`](../../data/README.md) 和 [`data/manifest.csv`](../../data/manifest.csv)。 |
| [EB-GAD（2026）](https://arxiv.org/abs/2609.38424)，training-free finite-horizon / empirical Bayes | 7 个标准图：Weibo、Reddit、Amazon、YelpChi、BlogCatalog、Facebook、ACM；4 个金融图：T-Finance、Elliptic、Elliptic++、DGraph。论文在 Amazon/YelpChi 还报告 TAM 小图上的评估；Elliptic/Elliptic++ 使用有标签交易子集；DGraph 在有标签节点上评估。 | 除 DGraphFin 外均已准备。Weibo 标签版本、Amazon/YelpChi 图大小、Elliptic 全量与 GGAD 过滤版需按论文协议选择。DGraphFin 需要官方免费账号注册，尚未下载。 |
| [Is Real-World Training Data Necessary for Generalist Graph Anomaly Detection?（AG-Forge + TS-GGAD，2026）](https://arxiv.org/abs/2610.12167)，合成训练图 + generalist GAD | Cora、Pubmed、Citeseer、CS、ACM、BlogCatalog、Flickr、Facebook、Amazon、Photo、YelpChi、Weibo、Questions、Reddit，共 14 个真实图。 | 14 个数据集都有本地候选版本。AlignGAD 作者仓库中的文件已按 `*-aligngad.mat` 单独留存；Weibo 是 PyGOD 版。论文边数可能按有向项统计，本仓库也有无向边数/稀疏矩阵 nnz，需统一口径。 |
| [AlignGAD（2026）](https://arxiv.org/abs/2606.12673)，zero-shot generalized GAD | 源图包括 Facebook、Flickr、BlogCatalog、ACM；目标图包括 Cora、CiteSeer、PubMed、Photo、CS、Amazon、YelpChi、Reddit。 | 作者仓库的数据副本已放到共享 `data/`，独立文件名以 `-aligngad.mat` 结尾；和其他 benchmark 版本并存，避免误覆盖。 |

## 新增或补全的数据文件

| 数据 | 本地路径 | 适合用途与核对事项 |
|---|---|---|
| Weibo | [`data/Weibo/`](../../data/Weibo/) | PyGOD 官方发布包已下载并解压。论文数据表的标签数有冲突：AG-Forge 表/PyGOD 说明列 868，EB-GAD 对其读取版本报告 347。正式实验前从实际载入的标签张量计算正类数，按文件哈希记录版本。 |
| Elliptic++ | [`data/EllipticPlusPlus/Transactions/`](../../data/EllipticPlusPlus/Transactions/) | 官方交易级 CSV，完整保存 `txs_features.csv`、`txs_classes.csv`、`txs_edgelist.csv`。EB-GAD 报告约 182 维；官方表可能把时间步另计为一个特征。 |
| Elliptic（标准全量） | [`data/Elliptic/elliptic-original-reconstructed/`](../../data/Elliptic/elliptic-original-reconstructed/) | 从 Elliptic++ CSV 使用 [EB-GAD 转换脚本](../../Repositories/EBGAD/tools/reconstruct_elliptic.py)生成的 166 维标准交易特征版本，203,769 节点、234,355 边。已有 `elliptic-ggad.mat` 是 46,564 节点的过滤图，不能代替全量版。 |
| Amazon 多关系 | [`data/Amazon/care-gnn/`](../../data/Amazon/care-gnn/) | CARE-GNN 作者仓库的 11,944 节点版本，保留压缩包和 `.mat`；与 NeighborDiv/AG-Forge 的 10,224 节点版本并存。 |
| YelpChi 多关系 | [`data/YelpChi/care-gnn/`](../../data/YelpChi/care-gnn/) | CARE-GNN 作者仓库的 45,954 节点版本，含 `rur/rtr/rsr/homo` 等关系；与 23,831 节点单关系版本并存。 |
| AlignGAD 数据副本 | `data/*/*-aligngad.mat` | 从本地 AlignGAD 作者仓库复制，保留来源后缀；严格复现实验前需对照其论文所报数据统计、标签和分割。 |
| DGraphFin | [`data/DGraphFin/README.md`](../../data/DGraphFin/README.md) | 尚无本地数据包。官方站点需要用户自行免费注册并获取，当前只登记获取步骤，不存账号、密码或登录令牌。 |

## 版本选择提示

1. **先按论文协议选文件，不按数据集名字选。** Amazon 至少有 10,224 节点 TAM/FreeGAD 版和 11,944 节点 CARE-GNN/GGAD 多关系版；YelpChi 至少有 23,831 节点版本和 45,954 节点版本。
2. **区分全图与评价子集。** EB-GAD 的 Elliptic 与 Elliptic++ 报告全量原始图规模，但部分指标在 46,564 个已标注交易节点上计算；DGraphFin 有 3.7M 全图和约 1.2M 已标注节点的评估口径。
3. **边数统计口径不同。** 有的论文列无向边，有的列 `edge_index` 的双向条目或多关系边总数。记录 `n_nodes`、原始边条数、对称化后稀疏矩阵 `nnz` 和自环处理，不能只抄一个“边数”。
4. **Weibo 标签数必须本地核对。** 当前两篇近期论文对 PyGOD 文件标签数的记载不一致；不要为了匹配论文结果手工改标签。
5. 数据分发和发表时保留原始来源、论文引用及许可说明。原始压缩包保留在对应目录，派生数据单独标注生成来源。

## 主要来源

- [NeighborDiv 论文](https://arxiv.org/abs/2605.20879)
- [EB-GAD 论文](https://arxiv.org/abs/2609.38424)；[作者代码及数据说明](../../Repositories/EBGAD/DATA.md)
- [AG-Forge / TS-GGAD 论文](https://arxiv.org/abs/2610.12167)
- [AlignGAD 论文](https://arxiv.org/abs/2606.12673)；[作者仓库](https://github.com/nhphan/AlignGAD)
- [PyGOD 官方数据仓库（Weibo）](https://github.com/pygod-team/data)
- [Elliptic++ 官方仓库](https://github.com/git-disl/EllipticPlusPlus)；[官方数据目录](https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l?usp=sharing)
- [CARE-GNN 官方代码及 Amazon/YelpChi 数据包](https://github.com/YingtongDou/CARE-GNN)
- [DGraphFin 官方数据页](https://dgraph.xinye.com/dataset)；[官方基线仓库](https://github.com/DGraphXinye/DGraphFin_baseline)
