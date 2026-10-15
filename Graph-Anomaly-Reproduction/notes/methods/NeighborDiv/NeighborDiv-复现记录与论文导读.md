# NeighborDiv 复现记录与论文导读

2026-10-12 更新：已按论文静态复检和修改代码，详见 [论文代码复检与后续验证](NeighborDiv-论文代码复检与后续验证.md)。随后按用户请求运行的验证、主实验和消融见 [验证运行记录](NeighborDiv-验证运行记录-2026-10-12.md)。下文 Cora 数值来自修改前的历史运行，不能作为当前代码验证结果。新版使用共享 `data/`、补齐 NPZ/DGL、阈值及实验入口，并记录数值后端的实现选择。

## 项目状态

- 论文：Kaifeng Wei et al., *NeighborDiv: Training-free Zero-shot Generalist Graph Anomaly Detection via Neighbor Diversity*, arXiv:2605.20879v1，2026-05-20。它目前是 arXiv 预印本。
- 论文 PDF：[NeighborDiv arXiv 预印本](../../../Papers/methods/NeighborDiv%20Training-free%20Zero-shot%20Generalist%20Graph%20Anomaly%20Detection%20via%20Neighbor%20Diversity%EF%BC%8C2026%20%E5%B9%B4%20arXiv%20%E9%A2%84%E5%8D%B0%E6%9C%AC.pdf)；全文提取文本位于本目录的 `NeighborDiv-paper-extracted.txt`。
- 代码：`Repositories/NeighborDiv/` 为依照论文方法节、实验设置和附录 B.4 独立编写的实现；本次搜索没有找到作者发布的实现仓库。
- 环境：WSL Ubuntu 20.04，沿用 `~/.venvs/cola`（Python 3.7.9、NumPy 1.21.6、SciPy 1.7.3、scikit-learn 1.0.2）。此实现不依赖 PyTorch、PyG 或 DGL。

## 算法和代码对照

1. 论文第 3.1 节：用截断 SVD 取 `r=8`，代码在 `src/neighbordiv.py` 的 `fit_score`，通过 `TruncatedSVD` 得到 `UΣ` 表示。
2. 对低维特征逐行做 L1 归一化，再做 L2 归一化以计算余弦相似度。
3. 论文第 3.2 节：对节点 i 的一跳邻居集合 `N(i)`，收集邻居两两之间的余弦相似度，取其方差 `D_i`。注意比较的是“邻居与邻居”，不是中心节点与邻居。
4. 论文第 3.3 节：用全图有效节点（度数至少为 2）的多样性中位数做参考，对 `|D_i - median(D)|` 做标准化，作为异常分数；度数小于 2 的节点得分为 0。
5. 论文主结果用全部无序邻居对；`--max-pairs 100/50` 是论文效率消融的抽样版本。
6. 标签不会进入打分函数；只在 `evaluate_scores` 中计算 AUROC、AP。

论文只写“variance”，没有说明总体方差还是无偏样本方差。当前实现使用总体方差（`ddof=0`），并在 `run.json` 记录此实现选择。新版分别固定 SVD 和 pair sampling 的种子；相同数据、版本和参数下的运行可重复性需后续验证。Full 仅表示邻居对不采样，不代表 SVD 后端没有随机性。

## 修改前已运行的首轮 Cora 结果（历史记录）

命令：

```bash
cd "/mnt/c/Users/28641/Desktop/论文科研-江权学长/Graph-Anomaly-Reproduction/Repositories/NeighborDiv"
~/.venvs/cola/bin/python scripts/run_neighbordiv.py \
  --data ../CoLA/dataset/cora.mat \
  --rank 8 --random-state 0 \
  --output-dir ../../results/NeighborDiv/cora-full
```

本次结果：AUROC `0.6672`，AP `0.0902`，SVD 和评分约 `1.30s`。同参数首轮运行约 `0.35s`；小图 CPU 单次计时会受系统负载影响。结果文件和参数记录在 `results/NeighborDiv/cora-full/`。

这次先验证独立实现、MAT 加载、分数和指标输出链路，不作为论文表格的严格复现。数据文件 `Repositories/CoLA/dataset/cora.mat` 有 2,708 个节点、1,433 维特征、150 个异常（5.53%），节点数和异常比例与论文 Table 4 相同；但文件有 11,606 个有向存储条目（对应 5,803 条无向边），论文 Table 4 写 5,429 条边。有效节点数本次是 2,239，论文附录报告 2,233。图版本/注入细节存在差异，所以当前 AUC/AP 不应直接与论文 Cora 分数比较。后续应找回或按论文所用数据处理流程重建其 Cora 文件，再比较。

## 文件位置

- 独立实现和命令：`Repositories/NeighborDiv/README.md`
- 算法主体：`Repositories/NeighborDiv/src/neighbordiv.py`
- CLI：`Repositories/NeighborDiv/scripts/run_neighbordiv.py`
- Cora 启动脚本：`Repositories/NeighborDiv/scripts/run_cora.sh`
- 分数：`results/NeighborDiv/cora-full/anomaly_scores.csv`
- 指标、输入 MAT 的 SHA-256、键名与运行参数：`results/NeighborDiv/cora-full/run.json`

## 建议的复现步骤

1. 逐段读 `fit_score` 并将输出维度与论文公式对应：`X (N×D)` → `UΣ (N×8)` → `D_i (N,)` → `score (N,)`。
2. 先固定数据哈希、SVD 随机种子和 `r=8`，运行 full-pair 主设置。
3. 确认标签仅用于评估，再对齐论文所用 Cora 图及注入版本。仅仅节点数/异常率一致不足以证明数据相同。
4. 对齐后比较 Cora AUROC/AP；再用 `--max-pairs 100`、`50` 和多个 `--random-state` 重现采样效率消融。
5. 做受控消融：SVD rank、variance/std/mean、median/mean 校准、低度节点回退。每次只改一项，并报告多次采样均值与标准差。

## 论文阅读顺序

1. Introduction：training-free 与 generalist GAD 的动机。
2. Section 3.1–3.3：公式 (1)–(13)，特别区分 Node-to-Neighbor consistency 与 Neighbor-to-Neighbor diversity。
3. Section 4.1–4.2：两种 GGAD 协议、七个测试集以及 Table 1–3。
4. Appendix B.1/B.2/B.4：数据统计、协议、实现参数。
5. Appendix D/E：启发式比较、方差/中位数消融、采样复杂度、低度节点和 SVD rank 敏感性。

## 已知风险与结论边界

- 论文是新近 arXiv 预印本，尚无同行评审会议/期刊版本可确认；论文自报的 SOTA 数值需要独立验证。
- 当前实现默认保留输入的有向边方向、去除自环并把非零边二值化。若复现数据本身是有向图，需对照论文数据预处理决定是否先对称化。
- 高度节点会使 full pair 计算的成本按 `choose(d_i, 2)` 增长；大图请先使用 `--max-pairs 100`，并把它明确记为近似版本。
- Cora 论文数据边数与现有注入文件不一致，是当前与论文数值差异的已知来源之一。
