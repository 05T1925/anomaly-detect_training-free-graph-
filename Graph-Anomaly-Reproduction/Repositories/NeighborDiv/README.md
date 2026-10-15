# NeighborDiv 论文实现

依据本地 **NeighborDiv: Training-free Zero-shot Generalist Graph Anomaly Detection via Neighbor Diversity**，arXiv:2605.20879v1 实现。论文位于 `../../Papers/methods/`。

**2026-10-13 状态：复检发现并修正 Disney 边处理、AMEN 归一化和部分节点计时掩码错误。** Disney 主 AUROC/AP 已对齐论文到四位小数。Cora 选择匹配 Table 16 的 FreeGAD 副本；Tolokers/Questions 的入度统计匹配 Table 16，因此增加入邻居候选协议，尚不能确认作者实际邻居构造。其他主指标仍有差距。七图主实验与受修改影响的消融保存在 `results/NeighborDiv/protocol-audit-2026-10-13/`；最新证据和剩余问题见 [`指标差距根因复检`](../../notes/methods/NeighborDiv/NeighborDiv-指标差距根因复检-2026-10-13.md)。[2026-10-12 记录](../../notes/methods/NeighborDiv/NeighborDiv-验证运行记录-2026-10-12.md) 是历史结果，不能据其已运行数量宣称复现完成。

复检证据、未明确的论文实现细节和下一位模型的验证清单见 [论文代码复检与后续验证](../../notes/methods/NeighborDiv/NeighborDiv-论文代码复检与后续验证.md)。该目录是按论文独立编写的实现。

## 方法与默认设置

最新[最终复检结论](../../notes/methods/NeighborDiv/NeighborDiv-最终复检结论-2026-10-13.md)与 `results/NeighborDiv/final-audit-2026-10-13/` 包含公开来源字节核验、七图独立评分/评价、44 个投影对照、49 个校准范围对照和 60 个 SBM 采样语义配对运行。本轮主评分代码未修改；诊断配置没有按标签替换为默认配置。

1. 第 3.1 节、式 (1)：`X -> U_r Sigma_r`，`r=8`，不中心化；先 L1 再 L2 行归一化。
2. 第 3.2 节、式 (2)-(9)：对每个节点的一跳邻居之间的无序节点对计算余弦相似度，再取总体方差。
3. 第 3.3 节、式 (10)-(13)：仅在 `degree>=2` 的节点上求多样性中位数、绝对偏差和 z-score；低度节点默认分数为 0。
4. 第 3.4 节、式 (14)-(15)：在全体节点分数上计算 `tau=mean(score)+lambda*std(score)`，默认 `lambda=1`，按严格 `score>tau` 输出二值结果。
5. AUROC/AP 只使用连续分数；标签只进入评估阶段，不进入特征投影、打分或阈值选择。

主结果采用 **Full：全部邻居对**，与附录 B.4 一致。`--pair-batch-size` 只控制分块内存，Full 会遍历所有块。`--max-pairs 100/50` 才是采样消融：每个节点最多均匀抽取 k 个无序邻居对，不放回。

可选 `--full-backend moments` 使用中心化 Gram 矩公式计算同一组全部邻居对的均值/总体方差，不做采样，计算量为每节点 `O(d*r^2)`。默认仍为 `stream` 逐对计算。七图独立核对的方差误差不超过约 `6e-16`；极近的浮点分数仍可能影响并列排序，T-Finance AP 相差约 `2.5e-7`。运行元数据分别记录逻辑覆盖的邻居对数和实际枚举的点积数。该后端耗时不能冒充论文 Table 14 的逐对算法耗时。

论文没有指定 SVD 求解器、随机种子、方差/std 的 ddof、偶数长度的 median 规则或零向量处理细节。当前用 float64、scikit-learn randomized SVD、`n_iter=5`、总体方差/std、NumPy 中位数；零投影向量贡献相似度 0，偏差标准差为 0 时有效节点也给 0 分并记录退化状态。上述选择可与作者实现存在差异，不应视为已确认的作者默认值。

## 文件结构

```text
src/data.py                       # MAT、Yandex NPZ、可选 DGL/ZIP 加载与输入校验
src/neighbordiv.py                 # 评分、分块/采样、阈值、评估
src/heuristic_baselines.py         # Appendix D 启发式基线；大图 AMEN 有 OpenMP 内核
src/amen_ego_native.cpp            # 与 Python AMEN 参考实现对照的位集内核
configs/paper_experiments.json     # 7 个目标图、论文参考值、数据版本说明
scripts/run_neighbordiv.py         # 单图入口
scripts/run_paper_experiments.py   # 实验计划 / 批量运行 / 汇总
scripts/run_appendix_d_heuristics.py  # Appendix D 的 LCC/NRS/PCD/AMEN-Ego
scripts/run_appendix_f_sbm.py        # Appendix F 的本地 SBM sweep
scripts/run_appendix_g_diagnostics.py # Appendix G 的 T-Finance 诊断
scripts/run_entropy_bin_sensitivity.py # Appendix E.1 的本地分箱敏感性
scripts/run_cora.sh                # 新 Cora 结果目录的 WSL 入口
```

核心依赖是 NumPy、SciPy、scikit-learn，见 `requirements.txt`。本轮常规验证和六个数据集实验在 WSL `~/.venvs/cola`（Python 3.7.9 / NumPy 1.21.6 / SciPy 1.7.3 / scikit-learn 1.0.2）运行；T-Finance 使用单独的 `~/.venvs/tfgad` 环境（Python 3.8.10 / PyTorch 2.1.0 CPU / DGL 1.1.3）。未安装或更改依赖。

## 单图复跑示例

以下命令可用于单独复跑 Disney。验证记录中的完整结果已经保存在另一目录；复跑时请使用新的空输出目录。

```bash
cd "/mnt/c/Users/28641/Desktop/论文科研-江权学长/Graph-Anomaly-Reproduction/Repositories/NeighborDiv"
~/.venvs/cola/bin/python scripts/run_neighbordiv.py \
  --data ../../data/Disney/disney-dominant.mat \
  --edge-policy undirected --rank 8 --svd-random-state 0 --pair-random-state 0 \
  --output-dir ../../results/NeighborDiv/disney-rerun-example
```

Cora 可使用 `bash scripts/run_cora.sh`，读取 FreeGAD 副本并输出 `results/NeighborDiv/cora-table16-audit-full/`。该副本对齐 Table 16，但作者输入文件与注入后边数仍待确认。

单次运行写入：

- `anomaly_scores.csv`：节点编号、连续分数，保留 17 位有效数字。
- `node_diagnostics.csv`：度数、有效节点标记、局部统计量、偏差、分数、二值结果。低度节点的局部统计量/偏差为 NaN，异常分数仍为回退值。
- `run.json`：数据和源码 SHA-256、数据处理、求解器/种子/统计约定、环境版本、指标、阈值与分阶段耗时。

默认要求输出目录为空。重复实验请选择新目录；仅在确实要替换时显式指定 `--overwrite`。

## 论文实验计划和批量运行

`run_paper_experiments.py` 默认只打印计划，不计算分数或写结果文件；执行实验必须添加 `--execute`。

```bash
# 只查看 Disney/Cora 主设置的计划
~/.venvs/cola/bin/python scripts/run_paper_experiments.py --suite main --datasets Disney Cora

# 运行两个目标图的主设置
~/.venvs/cola/bin/python scripts/run_paper_experiments.py \
  --suite main --datasets Disney Cora --execute

# 后续运行采样消融：Full 一次，k=100/50 各 5 次
~/.venvs/cola/bin/python scripts/run_paper_experiments.py \
  --suite sampling --datasets Cora Reddit --execute
```

可选 suite：

| suite | 对应设置 |
|---|---|
| `main` | 7 个目标图的 Full、r=8 主设置；不指定 datasets 时选择全部目标图 |
| `sampling` | Full、k=100、k=50；采样默认 seeds=0..4，SVD seed 固定 |
| `rank` | Cora/YelpChi/Reddit/Questions 的 r=8/16/32/64；YelpChi 不运行 r=64 |
| `low-degree` | zero fallback、median fallback、valid-only evaluation，对应 Table 17 |
| `statistic` | variance+median、variance+mean、std+median、mean+median；Table 11 的部分消融 |

默认结果根目录为 `../../results/NeighborDiv/paper-audit/`。各 suite 记录 `plan.json`、`summary.csv`、`summary.json`、`aggregate.json`。采样重复的标准差表示固定 SVD 下的采样随机性，不是 Table 2 的 source-domain 标准差。七目标图全部完成且指标有效时才汇总七图平均，子集不会冒充论文七图均值。

可通过 `--svd-seed`、`--svd-algorithm arpack`、`--seeds` 或 `--output-root` 明确设置一次新实验；不应根据测试标签搜索最优参数。论文未提供具体的采样 seed 值，0..4 是本地复现约定。

Appendix E.1 的熵统计敏感性可用 `scripts/run_entropy_bin_sensitivity.py` 运行；该脚本固定相似度区间 `[-1,1]`，比较等宽 5/10/20/50 箱，T-Finance 使用 k=100/50 各五个采样 seed。论文未说明熵的箱数和边界，所以这些结果只用于显示该选择对指标的影响，不作为 Table 11 的精确复现。

Appendix D/F/G 的结果与剩余限制见验证运行记录。训练式基线（例如 Appendix C 的跨源训练表格）没有在本地 NeighborDiv 实现中复跑；对应模型代码、各训练协议和完整检查点未纳入当前运行环境。Table 2 已给出的源域标准差没有额外重训，论文 Appendix G 也说明不增加训练随机种子实验。

## 数据处理

共享数据位于 `../../data/`，详见 [数据目录](../../data/README.md)。

- **MAT**：自动读取 `Network/A/adj/Adj`、`Attributes/X/features/Features`、`Label/gnd/labels/y`；可显式指定键。标签必须是 0/1，多类别或小数标签会报错。`--no-labels` 可以只计算分数。
- **Yandex NPZ**：读取 `node_features/node_labels/edges`；官方文件每条无向边存一次，所以 `--edge-policy auto` 为该格式补反向边；保留原始节点次序，忽略节点分类 split masks。[官方说明及加载器](https://github.com/yandex-research/heterophilous-graphs)
- **DGL**：`--format dgl` 支持无扩展名图文件；读取特征与标签，保留边方向。T-Finance 二列 one-hot 标签按官方 BWGNN 读取方式解码。[BWGNN 加载器](https://github.com/squareRoot3/Rethinking-Anomaly-Detection/blob/master/dataset.py)
- **T-Finance ZIP**：`--format dgl-zip` 只把 `tfinance` 成员流式解压到临时目录，读取后清理，不写入共享数据目录。运行时仍需临时磁盘空间和图加载内存。

全部格式合并重复邻接项、把非零边二值化、去除自环，记录原始/处理后邻接项数。MAT/DGL 的通用默认保留输入方向；论文批处理配置单独选择 Disney 的 `undirected`、Tolokers/Questions 的候选 `incoming`，其余保持 `preserve`。NPZ 通用 `auto` 仍采用官方双向并集；复跑本次候选协议时应使用批处理配置或显式指定参数。Table 16 的总数对齐不能证明邻居列表和特征文件逐项相同。不要把存储项数、自环项数和无向边数混为一谈。

本实现的 CPU 耗时及计时边界与作者硬件/PyTorch 实现不同，只可报告本机观察值。`load_and_scoring_seconds` 包含数据读取，`scoring_seconds` 仅包含特征投影和打分，均排除哈希计算与输出文件写入。
