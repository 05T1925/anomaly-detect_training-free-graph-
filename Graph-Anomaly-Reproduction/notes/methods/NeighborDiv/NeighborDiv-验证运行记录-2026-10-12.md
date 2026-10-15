# NeighborDiv 验证与实验运行记录

> **2026-10-13 更正：本页保留历史运行事实，部分解释已被后续复检推翻。** Disney 的差距来自本地边处理错误，修正后 AUROC/AP 与论文一致；Tolokers/Questions 的 Table 16 对应原始入度，不能把差异全部归因于数据版本。历史 AMEN 的内部归一化下界有公式错误；Python/C++ 相互一致未能发现共同错误。Cora 已找到匹配 Table 16 的 FreeGAD/AlignGAD 副本。原页的“已运行”不代表实验指标已复现，原七图均值和 AMEN 表格不得作为当前结果引用。详见 [2026-10-13 根因复检](NeighborDiv-指标差距根因复检-2026-10-13.md)。

执行日期：2026-10-12。按用户请求启动此前列出的验证清单；本记录与 `NeighborDiv-论文代码复检与后续验证.md` 中最初“尚未运行”的静态复检状态相衔接。

## 环境和结果位置

- 常规实验：WSL2 Ubuntu 20.04，Python 3.7.9、NumPy 1.21.6、SciPy 1.7.3、scikit-learn 1.0.2，使用 `~/.venvs/cola`。
- DGL/T-Finance：独立 `~/.venvs/tfgad` 环境，Python 3.8.10、NumPy 1.24.4、SciPy 1.10.1、scikit-learn 1.3.2、PyTorch 2.1.0+cpu、DGL 1.1.3。
- Full 主实验：[`results/NeighborDiv/paper-audit/main`](../../../results/NeighborDiv/paper-audit/main)。
- 六图采样：[`results/NeighborDiv/paper-audit-suites/sampling`](../../../results/NeighborDiv/paper-audit-suites/sampling)。
- Rank、低度节点和统计量消融：[`results/NeighborDiv/paper-audit`](../../../results/NeighborDiv/paper-audit) 下相应子目录。
- T-Finance 采样：[`results/NeighborDiv/paper-audit/sampling/T-Finance`](../../../results/NeighborDiv/paper-audit/sampling/T-Finance)。
- T-Finance Full：[`results/NeighborDiv/paper-audit-tfinance-full-retry/main`](../../../results/NeighborDiv/paper-audit-tfinance-full-retry/main)，后台日志为 [`results/NeighborDiv/tfinance-full-retry.log`](../../../results/NeighborDiv/tfinance-full-retry.log)。
- 无标签 CLI 检查：[`results/NeighborDiv/no-labels-audit`](../../../results/NeighborDiv/no-labels-audit)。

每次运行均单独保留 `run.json`、连续分数和节点诊断 CSV；批量套件另有计划、逐次汇总和聚合 JSON/CSV。旧 `results/NeighborDiv/cora-full/` 未覆盖。

## 正确性验证

| 检查 | 结果 |
|---|---|
| Python `compileall`、模块导入、批量实验 plan-only | 通过 |
| 手算小图：邻居相似度、总体方差 `1/9`、有效节点 median、z-score；度数 0/1/2 | 通过 |
| 邻居对 rank 对 `d=2..39` 的 `p<q` 一一映射；Floyd 抽样无重复、大小和固定 seed 可复现 | 通过 |
| Full 分块大小 1、2、3、99 的结果一致（浮点误差容差 `1e-13`） | 通过 |
| 零投影行、`1e-300` 微小非零行、零标准差退化校准 | 通过；分数有限，退化状态有记录 |
| 小数、多类别、长度不符标签拒绝；one-hot 标签解码；评估 mask、无标签评估 | 通过 |
| 合成与真实 Yandex NPZ：对称并集、去自环/重复边、原始节点顺序 | 通过 |
| 合成 DGL：边方向、去重/去自环、二列 one-hot 标签 | 通过 |
| T-Finance DGL ZIP：加载、one-hot 标签解码、临时解压读取 | 通过 |
| Disney `--no-labels` 完整 CLI 输出 | 通过；生成分数，AUROC/AP 按设计跳过 |
| 新实现 143 个已完成 `run.json`：源码哈希、指标有限性及两个 CSV 文件头 | 通过，0 个错误（不含修改前的历史 `cora-full`） |
| 143 个新实现输出：诊断 CSV 的节点有效标记和实际邻居对数逐节点重算对照 | 通过，0 个计数错误 |

手算小图曾发现 `rank == 特征维数` 时 scikit-learn `TruncatedSVD` 的限制与参数校验不一致。已在 [`src/neighbordiv.py`](../../../Repositories/NeighborDiv/src/neighbordiv.py) 增加精确全维 NumPy SVD 分支；小图边界验证和 YelpChi `r=32` rank sweep 均通过。

## Full 主实验

七个目标数据集均成功完成 Full（不抽样），固定 SVD seed 为 0、rank 为 8。表中有效节点按实现的 `degree >= 2` 统计。

| 数据集 | AUROC | AP | 有效节点：本次 / 论文 Table 16 | 评分秒数 |
|---|---:|---:|---:|---:|
| Cora | 0.667158 | 0.090236 | 2239 / 2233 | 1.12 |
| YelpChi | 0.581478 | 0.069071 | 15679 / 15679 | 0.99 |
| Reddit | 0.501402 | 0.035603 | 8332 / 8332 | 2.02 |
| Tolokers | 0.528608 | 0.219239 | 11339 / 9853 | 16.23 |
| Disney | 0.554379 | 0.129554 | 94 / 113 | 0.02 |
| Questions | 0.499223 | 0.033221 | 22931 / 13567 | 4.25 |
| T-Finance | 0.920228 | 0.531983 | 39275 / 39275 | 11095.82 |

T-Finance 有 39,357 个节点、42,445,086 个有向邻接项、39,275 个有效节点；最高度为 7,723，总 Full 工作量为 96,732,893,889 个邻居对。Full 成功完整计算这些邻居对，评分本身耗时 11,095.82 秒（3 小时 4 分 56 秒）；加载加评分共 11,139.46 秒（3 小时 5 分 39 秒），其中加载约 43.64 秒。第一轮前台执行运行约 49 分钟后会话以退出码 1 消失，没有 traceback、结果或批次汇总；计划保留在 `results/NeighborDiv/paper-audit-tfinance-full/main/`。脱离前台会话的重跑成功，完整产物在 [`results/NeighborDiv/paper-audit-tfinance-full-retry/main/T-Finance/full/seed-0`](../../../results/NeighborDiv/paper-audit-tfinance-full-retry/main/T-Finance/full/seed-0)，汇总在同目录下的 `summary.csv`，运行日志为 `results/NeighborDiv/tfinance-full-retry.log`。

七图 Full 结果的描述性算术平均为 AUROC `0.607496`、AP `0.158415`；计算时合并了六图主套件与单独保存的 T-Finance Full 汇总。默认 `paper-audit/main/aggregate.json` 因只对应六图套件，仍记录六图聚合。由于 Cora、Disney、Tolokers、Questions 的共享数据拓扑/有效节点数与论文统计不同，该七图平均值是当前数据版本的汇总，不是严格的作者数据复现值。

输入拓扑统计存在不能忽略的差异：Cora 为 5,803 条无向边、有效节点数多 6；Disney 有效节点少 19；Tolokers 有效节点多 1,486；Questions 有效节点多 9,364。指标按现有共享数据原样计算，不据 AUC 修改边或标签；论文值只作为事后参照。YelpChi 和 Reddit 的有效节点数与配置中的论文统计一致。

## 采样与消融

- 六个非 T-Finance 目标图：采样套件 `66/66` 成功，其中每图 Full 一次、`k=100` 五个 seed、`k=50` 五个 seed；SVD 固定 seed=0，pair seed=0..4。
- T-Finance：`k=100` 和 `k=50` 各五个 seed，共 `10/10` 成功。运行环境为 CPU；每次评分约 9.8–14.7 秒（另有 ZIP 读取和 DGL 加载时间）。因邻居对总量约 968 亿，采样结果不代替 Full。
- T-Finance 固定 SVD seed 下的五 seed 均值（总体标准差）：`k=100` AUROC `0.903883 ± 0.000807`、AP `0.482607 ± 0.005662`；`k=50` AUROC `0.884844 ± 0.003080`、AP `0.430330 ± 0.003590`。
- Rank sweep：`15/15` 成功，覆盖 Cora、YelpChi、Reddit、Questions 的论文允许 rank；YelpChi 因特征维数为 32 不运行 `r=64`。
- 低度策略消融：`18/18` 成功，六图各运行 zero、median、valid-only。
- 统计量消融：`24/24` 成功，六图各运行 variance-median、variance-mean、std-median、mean-median。

上述核心套件均未报告运行失败。首轮记录时，Entropy、Appendix D/F/G 尚未运行；后续补充的实验及仍受论文协议限制的项目见下节。

运行中出现 SciPy `SparseEfficiencyWarning`（CSR 对角线处理）和全零特征合成用例中的 scikit-learn explained-variance `RuntimeWarning`；对应实验仍完成，评分输出有限。警告和实现约定应与结果一起保留。

## 后续补充实验（2026-10-12）

用户要求继续执行未跑的相关实验后，完成了 Appendix D/F/G，并对论文未给出分箱定义的 entropy 做了本地敏感性分析。下列结果不覆盖旧 `cora-full`，也不用于按标签调参。

### Appendix D：四种结构/属性启发式

LCC、NRS、PCD、AMEN-Ego 在七图上共 `28/28` 组合完成。逐数据集指标、节点数、耗时和输入哈希见 [`metrics.csv`](../../../results/NeighborDiv/paper-audit-appendix-d/metrics.csv)；分数和标签 CSV 见同目录。当前共享数据的七图算术平均与论文 Table 12 对照：

| 方法 | AUROC：本地 / 论文 | AP：本地 / 论文 |
|---|---:|---:|
| LCC | 0.5646 / 0.5775 | 0.0840 / 0.0892 |
| NRS | 0.5234 / 0.5402 | 0.0858 / 0.0823 |
| PCD | 0.5451 / 0.5378 | 0.0827 / 0.0787 |
| AMEN-Ego | 0.5726 / 0.5770 | 0.0925 / 0.0988 |

AMEN-Ego 按论文的 (I_{max}=|C|^2) 约定计算，并对 T-Finance 使用 OpenMP 位集内核；该内核与 Python 参考实现逐节点对照，Disney 最大绝对分差为 `5.6e-16`，含孤立点的合成图对照误差为 `0`。T-Finance AMEN-Ego 评分耗时 `131.70 s`，AUROC/AP=`0.674491/0.084489`。Table 12 的单数据集结果仍不完全一致，例如 Disney AMEN-Ego=`0.4958/0.0796`，论文为 `0.6123/0.0875`；所以只能报告为按论文定义的本地基线运行，不能宣称逐项数值复现。

### Appendix F：SBM 同配性 sweep

`5` 个 homophily 值 × `3` 个生成 seed × `3` 种异常类型，共 `45/45` 运行完成。参数、局部图生成/重连约定及完整逐次结果见 [`metadata.json`](../../../results/NeighborDiv/paper-audit-appendix-f/sbm/metadata.json)、[`runs.csv`](../../../results/NeighborDiv/paper-audit-appendix-f/sbm/runs.csv) 和 [`summary.csv`](../../../results/NeighborDiv/paper-audit-appendix-f/sbm/summary.csv)。论文未公布三个随机 seed 的数值；本地使用 `0,1,2`，并按明确记录的 expected-degree 与重连选择生成图。

| 类型 | homophily 0.1 / 0.3 / 0.5 / 0.7 / 0.9：本地 NeighborDiv AUROC | 论文 Table 18 |
|---|---|---|
| Type-H | 0.9991 / 1.0000 / 0.9997 / 0.9925 / 0.8355 | 0.998 / 0.995 / 0.978 / 0.933 / 0.799 |
| Type-D | 0.5134 / 0.4555 / 0.6500 / 0.6540 / 0.1759 | 0.785 / 0.778 / 0.880 / 0.847 / 0.362 |
| Mixed | 0.7584 / 0.7006 / 0.8515 / 0.8154 / 0.5327 | 0.859 / 0.864 / 0.915 / 0.876 / 0.522 |

差异较大，尤其 Type-D 与低/中 homophily 的 Mixed 结果；论文没披露所有图生成和注入细节，因此这组是已执行的本地 SBM 实验，不是 Table 18 数值复现。

### Appendix G：T-Finance 事后诊断

Full 与 k=100/k=50 五个 seed 的 AUROC/AP、PR 点、统计摘要见 [`metrics.csv`](../../../results/NeighborDiv/paper-audit-appendix-g/T-Finance/metrics.csv)、[`precision_recall_points.csv`](../../../results/NeighborDiv/paper-audit-appendix-g/T-Finance/precision_recall_points.csv) 和 [`summary.json`](../../../results/NeighborDiv/paper-audit-appendix-g/T-Finance/summary.json)。两张 SVG 均通过 XML 解析：[`PR 曲线`](../../../results/NeighborDiv/paper-audit-appendix-g/T-Finance/tfinance_pr_curve.svg)、[`分数分布`](../../../results/NeighborDiv/paper-audit-appendix-g/T-Finance/tfinance_score_distribution.svg)。

本地 Full 的 P@100/500/1000/5000=`0.390/0.720/0.684/0.3028`；论文 Appendix G 为 `0.580/0.830/0.804/0.295`。正常节点分数均值/总体标准差=`-0.1186/0.8018`，异常节点=`2.4682/1.4155`；KS=`0.7549`。SciPy 的 KS p 值浮点计算下溢为 `0.0`，摘要已注明这不代表精确 p 值字面为零。当前 ZIP 标签是 `1,804` 个正例，论文写 `1,803`，数据版本也不完全相同。

### Entropy 分箱敏感性（Table 11 的本地探索）

论文未给 entropy 箱数、相似度区间或边界规则，故固定区间 `[-1,1]` 并比较等宽 `5/10/20/50` 箱：Cora、YelpChi、Reddit、Tolokers、Disney、Questions 使用 Full；T-Finance 使用 k=100 和 k=50 各五个 seed。共 `64` 行指标，见 [`metrics.csv`](../../../results/NeighborDiv/paper-audit-entropy-bin-sensitivity/metrics.csv)、[`aggregate.csv`](../../../results/NeighborDiv/paper-audit-entropy-bin-sensitivity/aggregate.csv) 和 [`metadata.json`](../../../results/NeighborDiv/paper-audit-entropy-bin-sensitivity/metadata.json)。

箱数对结果影响明显。例如 Cora 的 AUROC 从 5 箱的 `0.5324` 变为 50 箱的 `0.7894`；T-Finance k=100 的均值从 `0.8461` 变为 `0.6071`。这确认熵消融高度依赖未披露的分箱选择；T-Finance 还是采样值，因此不能把此 sweep 与论文 Table 11 的七图 Full 平均直接比较。

### 仍未能复跑的论文结果

- Appendix C 中的学习式跨源训练表格，以及 Appendix G 的额外基线重训，需要对应方法的完整作者代码、训练配置/检查点及协议；当前 NeighborDiv 环境没有这些可直接运行的基线实现。论文 Appendix G 本身也明确没有为学习式基线增加随机种子训练，Table 2 的 source-domain 标准差是已有表格数据，不在本轮另行重训。
- Entropy 的 exact Table 11 结果仍不可确认，因为箱数与边界未披露；已跑的只是上述明确参数的敏感性实验。
- Appendix F seed 与生成/重连细节未公开；所有 45 个本地设置都已跑完，但结果不可逐数值归因到作者随机实例。

因此，用户要求中**所有能从当前代码和公开设置执行的新增实验均已完成**；以上协议/代码缺失项已实测到可复现边界并单独标记，没有以局部样本或本地选择冒充精确复现。
