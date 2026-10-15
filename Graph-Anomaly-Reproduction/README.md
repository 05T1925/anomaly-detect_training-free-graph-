# Graph Anomaly Reproduction

图异常检测论文学习与复现工作区。各论文仓库依赖不同，建议分别使用独立环境；数据集集中放在 `data/`，供多个实验共享。

## 目录导航

- [`Papers/README.md`](Papers/README.md)：论文资料目录索引，按方法论文、综述、工具及相关资料分类。
- `Repositories/`：公开代码仓库，以及 NeighborDiv 的独立论文实现。
- [`data/README.md`](data/README.md)：共享数据集说明；[`data/manifest.csv`](data/manifest.csv) 记录数据文件清单，SHA-256 见 `data/checksums.sha256`。
- [`论文—数据集对应与准备状态`](notes/references/paper-dataset-crosswalk.md)：NeighborDiv、EB-GAD、AG-Forge/TS-GGAD 等论文使用的数据集与本地版本、待办下载项。
- [`notes/README.md`](notes/README.md)：按方法归档的论文导读、代码讲解、实验记录和来源核验。
- `results/`：实验输出与指标。

## 方法、论文与学习记录

NeighborDiv 最新检查见 [最终复检结论](notes/methods/NeighborDiv/NeighborDiv-最终复检结论-2026-10-13.md)：官方 PDF、六个公开数据文件及本地复制/解压均通过核验，独立评分与评价复算支持当前实现；Questions 的校准范围和 SBM 的社区采样语义是新增的协议线索。Disney 主指标对齐，其余仍有差距。

| 方法 / 项目 | 代码目录 | 本地论文 | 学习与复现记录 |
|---|---|---|---|
| TFGAD | `Repositories/TFGAD/` | [`TFGAD-WWW-2025.pdf`](Papers/methods/TFGAD-WWW-2025.pdf) | [`首次复现与数据文件查看`](notes/methods/TFGAD/TFGAD-首次复现与数据文件查看.md)、[`实验记录`](notes/experiments/reproduction-log.md) |
| FreeGAD | `Repositories/FreeGAD/` | [`FreeGAD-CIKM-2025.pdf`](Papers/methods/FreeGAD-CIKM-2025.pdf) | [`复现记录与论文导读`](notes/methods/FreeGAD/FreeGAD-复现记录与论文导读.md) |
| CoLA | `Repositories/CoLA/` | [`CoLA-TNNLS-2022.pdf`](Papers/methods/CoLA-TNNLS-2022.pdf) | [`复现记录`](notes/methods/CoLA/CoLA-复现记录与论文导读.md)、[`逐段代码导读`](notes/methods/CoLA/CoLA-详细复现操作与代码逐段导读.md) |
| DOMINANT | `Repositories/DOMINANT/` | 尚未取得可合法公开下载的原论文 PDF；来源情况见 [`来源核验`](notes/references/paper-and-code-sources.md) | [`复现记录与论文导读`](notes/methods/DOMINANT/DOMINANT-复现记录与论文导读.md) |
| NeighborDiv（自实现） | `Repositories/NeighborDiv/` | [`arXiv 预印本 PDF`](Papers/methods/NeighborDiv%20Training-free%20Zero-shot%20Generalist%20Graph%20Anomaly%20Detection%20via%20Neighbor%20Diversity%EF%BC%8C2026%20%E5%B9%B4%20arXiv%20%E9%A2%84%E5%8D%B0%E6%9C%AC.pdf) | [`复现记录与论文导读`](notes/methods/NeighborDiv/NeighborDiv-复现记录与论文导读.md)、[`验证运行记录`](notes/methods/NeighborDiv/NeighborDiv-验证运行记录-2026-10-12.md) |
| EB-GAD | `Repositories/EBGAD/` | [`arXiv 全文 PDF`](Papers/methods/EB-GAD-arXiv-2609.38424.pdf) | 官方代码已公开；论文表格脚本见仓库 `neurips2026/` |
| AG-Forge / TS-GGAD | 暂无已确认的作者代码仓库 | [`arXiv 全文 PDF`](Papers/methods/AG-Forge-TS-GGAD-arXiv-2610.12167.pdf) | 有论文和对照结果；适合作为从论文独立实现的候选 |
| Federated GAD（Disentangled Representation） | 暂无已确认的作者代码仓库 | [OpenReview 全文](https://openreview.net/pdf?id=xqjnhRqdK9)；本地保存受 403 限制 | 联邦训练方向；与 training-free 主线不同 |
| SmoothGNN | `Repositories/SmoothGNN/` | [`arXiv 全文 PDF`](Papers/methods/SmoothGNN-arXiv-2405.17525.pdf) | 作者公开实现；尚未纳入本项目的复现实验 |
| AlignGAD | `Repositories/AlignGAD/` | [`arXiv 全文 PDF`](Papers/methods/AlignGAD-arXiv-2606.12673.pdf) | 作者公开仓库；尚未纳入本项目的复现实验 |
| ARC-zero | `Repositories/ARC-zero/` | [`arXiv 全文 PDF`](Papers/methods/ARC-zero-arXiv-2602.18793.pdf) | 作者公开实现；ARC few-shot 代码在另一个仓库 |
| PyGOD（基线工具库） | `Repositories/PyGOD/` | [`PyGOD-JMLR-2024.pdf`](Papers/tools/PyGOD-JMLR-2024.pdf) | 来源与使用范围见 [`来源核验`](notes/references/paper-and-code-sources.md) |

PyGOD 是包含多个检测器的工具库，不是单一模型。用它运行 CoLA 或 DOMINANT 时，应在实验记录中注明实际使用的是 PyGOD 实现。

推荐论文、全文、本地文件、作者仓库与代码缺失状态的一一对应见 [`论文—代码对应索引`](notes/references/paper-and-code-sources.md)。

NeighborDiv 目录是依据论文公式编写的独立实现。**2026-10-13 复检确认 Disney 边处理和 AMEN 归一化存在本地错误，已修正并重算；Disney 主指标已对齐论文到四位小数。** Cora 改用匹配 Table 16 的 FreeGAD 副本；Tolokers/Questions 增加入邻居协议，匹配论文度数统计，但这仍是待作者确认的协议推断，指标尚未对齐。已验证并提供可选 Full 矩公式后端，显著缩短重复消融的计算时间。最新结论与证据见 [`指标差距根因复检`](notes/methods/NeighborDiv/NeighborDiv-指标差距根因复检-2026-10-13.md)。2026-10-12 的 [`验证运行记录`](notes/methods/NeighborDiv/NeighborDiv-验证运行记录-2026-10-12.md) 和历史输出保留；运行结束不等于论文指标复现完成。

## 复现记录约定

记录数据来源及版本、预处理、超参数、随机种子、运行命令、指标和耗时。同名数据集不一定采用相同标签或异常注入协议；严格复现时先按论文指定来源准备数据，不要未经核对替换成其他仓库中的同名文件。
