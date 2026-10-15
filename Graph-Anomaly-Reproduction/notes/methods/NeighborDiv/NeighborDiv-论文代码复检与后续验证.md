# NeighborDiv 论文代码复检与后续验证

> 最新复检见 [2026-10-13 指标差距根因复检](NeighborDiv-指标差距根因复检-2026-10-13.md)。本页是历史静态检查；Disney 边处理、AMEN 下界公式、Cora 副本及 NPZ 度数协议已有后续修正。

复检日期：2026-10-12。依据本地 arXiv:2605.20879v1 PDF（29 页）。本文页码是 PDF 页码，也与该文件印刷页码一致。

**以上状态描述的是本静态复检完成时的状态。** 后续用户明确要求启动验证；核心实验与代码验证已完成，并继续补跑 Appendix D/F/G 和 entropy 分箱敏感性。最新结果和限制见 [NeighborDiv 验证运行记录](NeighborDiv-验证运行记录-2026-10-12.md)。下方覆盖表是运行前交接记录，不能代表当前状态。原来的 Cora AUROC 0.6672、AP 0.0902 来自旧实现。

以上是静态复检完成时的状态。用户随后明确启动了验证；实际运行状态和结果见 [NeighborDiv 验证运行记录](NeighborDiv-验证运行记录-2026-10-12.md)。

论文：[本地 PDF](<../../../Papers/methods/NeighborDiv Training-free Zero-shot Generalist Graph Anomaly Detection via Neighbor Diversity，2026 年 arXiv 预印本.pdf>)。实现：[代码 README](../../../Repositories/NeighborDiv/README.md)。参数与论文主结果：[paper_experiments.json](../../../Repositories/NeighborDiv/configs/paper_experiments.json)。

## 公式和代码逐项核对

| 论文位置 | 要求 | 当前代码 |
|---|---|---|
| p.4 §3.1，式 (1) | 对原始 X 截断 SVD，使用 U_r Sigma_r，r=8；L1 行归一化 | `src/neighbordiv.py::score_graph`，scikit-learn TruncatedSVD，默认不中心化、不做标签相关预处理 |
| p.4 §3.2，式 (2)-(4) | 行邻居 N(i)；比较邻居与邻居的余弦相似度 | CSR 行邻居；`_normalize_projected` 做 L1/L2，`_similarity_batches` 计算邻居对内积 |
| p.5，式 (5)-(9) | p<q 的无序邻居对；采样时均匀不放回 | Full 按 pair rank 分块遍历；Sampling 用 Floyd 抽取 pair rank 再反解端点 |
| p.5-6，式 (10)-(13) | degree>=2 为有效集；在有效集上 median、绝对偏差、均值/std 标准化 | `score_graph` 只用 valid 集校准；degree<2 默认零分，不混入校准统计 |
| p.6 §3.4，式 (14)-(15) | 对全体分数求 tau=mu+lambda*sigma，严格 s>tau，lambda=1 | `predict_anomalies`；阈值与标签无关 |
| p.17 B.4 | 主结果 Full；AUROC/AP 根据连续分数；没有训练、模型选择或 early stopping | CLI 默认 `max_pairs=None`，评估在评分完成后；无训练循环 |
| p.9 §4.3、p.24 Table 13/14 | k=100/50；每个采样版本 5 seeds；报告精度与耗时 | `run_paper_experiments.py --suite sampling`；SVD seed 固定，pair seeds 0..4 |
| p.24 Table 15 | r=8/16/32/64；YelpChi 的 d=32 不能运行 r=64 | `--suite rank`；保留 YelpChi r=32 的全特征维度设置 |
| p.25 E.5、p.26 Table 17 | zero fallback、median fallback、valid-only | `--suite low-degree`；valid-only 只限制评估集，校准仍按有效集 |

`fit_score` 保留旧 `(scores, metadata)` API；新增 `score_graph` 返回分数、局部统计、偏差、度数和 valid 标记，方便逐节点调试。算法 API 不接受 labels。

## 已修正和补齐的内容

**1. Full 模式的内存开销。** 旧实现为一个节点一次性分配 `np.triu_indices(d)` 和两个 `(choose(d,2),r)` 数组。p.17 B.4 要求 Full，而 p.23-24 的效率分析特别指出高程度节点昂贵。现按最多 65,536 对一批遍历全部 pair rank，用分块中心矩合并计算方差；分块不会变成近似采样。旧实现的评分逻辑基本对应公式，这里修正的是执行大图时的内存需求。Full 的计算总量仍随邻居对数增长，尚未实测本机耗时。

**2. 采样的内存与随机性记录。** 旧采样会构建 degree 大小的 row-start 表，且依赖 `rng.choice` 的内部策略。现直接用 Floyd 方法均匀不放回选 k 个 pair rank，以整数二分反解端点。存储随 k 增长，反解成本为 O(k log d)。固定 SVD 种子、单独改变 pair seed，避免把投影变化混入采样消融；这属于本地实验隔离约定，论文没有给出具体 seed 值。

**3. 标签校验。** 旧 MAT 加载器在验证前把标签转成 int64，小数可能被截断；评估未明确排除多类别标签。现先要求一节点一个 0/1 标签，再转整数。DGL 的 T-Finance 标签允许经过严格检查的二列 one-hot 格式。缺标签时可用 `--no-labels` 只评分，评估子集缺任一类别时记录指标未定义，不写出伪数值。

**4. 微小分数与退化图。** 旧代码在偏差 std 不大于 float64 epsilon 时直接全给 0，会丢掉小但非零的信号。现仅在 std 真正为 0 时回退，并在 run.json 标记。行归一化也只把真正零向量视为零，通过先按行最大绝对值缩放避免范数溢出/下溢。论文没有规定零向量余弦或零分母行为，此处是明确记录的工程约定。

**5. 阈值和诊断输出。** 补上 p.6 §3.4 二值决策；阈值计算范围是全体节点，校准范围才是 valid 集。新增 `node_diagnostics.csv`，把低度节点未定义的局部统计量保留为 NaN，把连续分数保留为指定回退值。AUROC/AP 继续直接使用连续分数。

**6. 数据格式和七个目标图。** 增加 Yandex NPZ 与可选 DGL/ZIP 读取。已只检查 NPZ 文件头：Tolokers 的 feature/label/edge shape 为 `(11758,10)/(11758,)/(519000,2)`，Questions 为 `(48921,301)/(48921,)/(153540,2)`。T-Finance ZIP 文件目录记录一个 `tfinance` 成员、682,904,644 字节，本轮未解压或加载它。

Yandex 官方明确说明每条无向边只存一次，其加载器对非 directed 版本补反向边。因此本加载器对该发布格式默认取邻接与转置的并集；直接按边列表单方向读取会改变邻居集合。[官方 README](https://github.com/yandex-research/heterophilous-graphs#datasets)、[官方 datasets.py](https://github.com/yandex-research/heterophilous-graphs/blob/main/datasets.py)。T-Finance 原始二列标签按 [BWGNN 官方加载器](https://github.com/squareRoot3/Rethinking-Anomaly-Detection/blob/master/dataset.py) 的 argmax 语义解码，先验证 one-hot；未沿用其中修改异常属性/标签的可选实验分支。

**7. 实验记录和输出保护。** 新目录 `cora-audit-full` 与批量目录 `paper-audit/<suite>/...` 区分修改前结果。记录数据/源码哈希、实际种子、求解器、环境版本、边处理和阶段计时，默认拒绝写入非空输出目录。批量失败会写明 error，不能当成完成实验；只有七个目标图全部完成时才给七图均值。

## 静态复检时的实验覆盖范围（启动运行前）

| 实验 | 2026-10-12 静态复检时状态 |
|---|---|
| NeighborDiv 主算法、7 target graph 的主配置 | 代码/配置已准备；均未运行新实现 |
| Tables 13/14：Full、k=100、k=50 | 入口已准备；5 个采样 seed 是本地指定 |
| Table 15：rank sweep | 入口已准备，按论文排除 YelpChi r=64 |
| Table 17：低度节点策略 | 入口已准备 |
| Table 11：variance/std/mean，median/mean | 部分入口已准备；entropy 未实现，因为分箱边界和数量不足以确认 |
| Appendix D/Table 12 的 LCC、NRS、PCD、AMEN-Ego | 本轮未实现 |
| Appendix F 的 SBM sweep、Appendix G 的完整诊断与图表 | 本轮未实现 |
| 训练式基线、SDIT/UMDT 跨源训练 | 本轮未实现；NeighborDiv 本身不使用 source graph，也不需重复四次训练 |

不能把“入口已经写好”当成“结果已经复现”，也不能把仅运行几个图的均值写成七图平均。

## 仍未对齐或论文未明确的事项

1. **Cora 版本差异已经存在。** 旧 run.json 的输入含 2,708 节点、150 异常、5,803 无向边、2,239 有效节点；论文 p.15 Table 4 为 5,429 边，p.25 Table 16 为 2,233 有效节点。共享 Cora 文件来自相同的 CoLA 版本。新代码不能凭空恢复作者的注入图；应先核对数据 SHA-256 和 Table 16，再解释指标差距。
2. **方差、std 和 median 约定未知。** 论文只用 Var、Std、Median 表述，B.4 没有公开 ddof 或调用代码。当前总体方差让只有一对邻居的 degree=2 节点有有限方差 0；这是实现选择。偶数长度 NumPy median 使用中间两项平均，也应记录。不能因作者使用 PyTorch 就推断其所有统计调用默认值。
3. **SVD 算法/精度未知。** 论文说明 truncated SVD，但未给求解器、迭代次数、oversampling、随机种子或 dtype。当前 randomized SVD 有固定 seed，也提供 ARPACK 选项；训练免除不等于任意数值后端都会给相同分数。应报告版本，不能挑 seed 追论文数值。
4. **MAT/DGL 的方向、自环和去重协议未知。** p.4 式 (2) 使用行邻居，没有逐数据集公布图转换。当前去自环、二值化、去重，并默认保留 MAT/DGL 方向；Yandex NPZ 按发布协议补反向。后续首先核对 Table 16，不要为了匹配 AUC 悄悄更改拓扑。
5. **运行时间不可直接对齐。** 作者用 PyTorch，当前为 NumPy/SciPy CPU，分块还增加了端点反解成本；论文 Table 14 没有提供充分的同硬件同计时边界信息。记录 loading/scoring/evaluation 分段时间，在本机比较 Full 与采样；不声称复现 63× 加速。
6. **表格参考值只用于事后比较。** 当前配置的论文 AUC/AP 来自 p.7 Table 1、p.17 Table 5；没有用参考值或测试标签决定参数。Table 2 的零 std 表示不依赖 source graph，不证明改变 SVD seed 后仍为零。

## 静态复检时整理的验证任务

以下是当时给下一位模型的交接清单。用户随后已明确要求执行；完成状态和结果以 [验证运行记录](NeighborDiv-验证运行记录-2026-10-12.md) 为准。

1. 先阅读本复检记录和代码 README，确认当前 WSL Python/NumPy/SciPy/scikit-learn 版本能导入实现；新代码未经过语法编译或模块导入检查，这些也留到该阶段。DGL 仅在 T-Finance 步骤检查，不要把所有环境合并。
2. 设计独立的小图手算对照：明确给定低维特征，人工列出邻居对余弦、总体方差、中位数、偏差、z-score 和阈值。覆盖 degree=0/1/2、零特征、全部偏差相同、极小但非零输入。不要只拿新实现自己的输出当期望值。
3. 单独验证 pair rank 与 p<q 的一一对应、样本没有重复、样本数量为 min(k,M)；比较 Full 分块与直接穷举小图的结果。在固定 SVD 下改变 pair_batch_size，分数应只有浮点舍入差异。
4. 验证异常标签小数、多类别、长度不符会报错；确认随机更换标签只会改变指标，不会改变评分/阈值。评估 mask 仅影响指标。无标签运行应正常保存分数。
5. 验证 NPZ 对称并集后的 node order、degree、自环/重复边处理；DGL 检查 one-hot 解码与边方向。首先比较论文 Table 16 的有效节点数，不满足时记录数据/处理差异。
6. 先运行最小 Disney Full，再 Cora Full，均使用新输出目录。检查分数有限、诊断 CSV/JSON 完整、输入/源码哈希、参数与实际代码对应。旧 Cora 数值仅是历史参考。
7. 基础正确性确认后扩展 YelpChi/Reddit/Tolokers/Questions；大图 T-Finance 先采样，再视机器资源决定 Full。不能把 k=100 结果冒充 Full；把原始文件版本与临时解压耗时写进记录。
8. 采样用固定 SVD seed 和 5 个独立 pair seed 汇总均值/std；之后再进行 rank、低度节点及统计量消融。填写实际结果与失败原因，不承诺一定接近论文指标。

后续可以直接向模型发：

> 阅读 NeighborDiv README 和“论文代码复检与后续验证”文件，现在允许开始测试。先做独立小图与采样/分块验证，再跑 Disney 和 Cora；保留旧结果，所有新实验写入新的 results 子目录。先报告数据/公式的一致性，再报告 AUROC/AP，不用标签调参追论文数值。
