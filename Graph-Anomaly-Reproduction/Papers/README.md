# Papers 论文资料索引

此目录按资料类型归档。本地 PDF 可直接打开；每篇论文的代码对应关系、获取来源和无法访问的情况见 [`notes/references/paper-and-code-sources.md`](../notes/references/paper-and-code-sources.md)，近期论文的数据集与本地版本见 [`论文—数据集对应表`](../notes/references/paper-dataset-crosswalk.md)。文件名沿用下载时的名字，方便与笔记和引用对应。

## 方法论文

| 文件 | 主题 |
|---|---|
| [`TFGAD-WWW-2025.pdf`](methods/TFGAD-WWW-2025.pdf) | Training-free、基于奇异值分解的图异常检测 |
| [`NeighborDiv` 预印本](<methods/NeighborDiv Training-free Zero-shot Generalist Graph Anomaly Detection via Neighbor Diversity，2026 年 arXiv 预印本.pdf>) | Training-free、zero-shot、基于邻居多样性的图异常检测；arXiv 预印本 |
| [`FreeGAD-CIKM-2025.pdf`](methods/FreeGAD-CIKM-2025.pdf) | FreeGAD 方法论文 |
| [`CoLA-TNNLS-2022.pdf`](methods/CoLA-TNNLS-2022.pdf) | 基于局部子图对比学习的图异常检测 |
| [`ANEMONE-CIKM-2021.pdf`](methods/ANEMONE-CIKM-2021.pdf) | 基于多视角对比学习的图异常检测 |
| [`EB-GAD-arXiv-2609.38424.pdf`](methods/EB-GAD-arXiv-2609.38424.pdf) | training-free，经验贝叶斯与有限时域图谱评分；官方实现见 `Repositories/EBGAD/` |
| [`AG-Forge-TS-GGAD-arXiv-2610.12167.pdf`](methods/AG-Forge-TS-GGAD-arXiv-2610.12167.pdf) | 合成异常图生成与 generalist GAD；目前未找到已确认的作者代码仓库 |
| [`SmoothGNN-arXiv-2405.17525.pdf`](methods/SmoothGNN-arXiv-2405.17525.pdf) | 无监督节点异常检测；作者实现见 `Repositories/SmoothGNN/` |
| [`AlignGAD-arXiv-2606.12673.pdf`](methods/AlignGAD-arXiv-2606.12673.pdf) | zero-shot 跨域图异常检测；作者实现见 `Repositories/AlignGAD/` |
| [`ARC-zero-arXiv-2602.18793.pdf`](methods/ARC-zero-arXiv-2602.18793.pdf) | zero-shot generalist GAD；作者实现见 `Repositories/ARC-zero/` |
| [`DOMINANT-SDM-2019-source.md`](methods/DOMINANT-SDM-2019-source.md) | DOMINANT 论文来源记录；目前目录中没有原论文 PDF |

## 可在线阅读全文、暂未保存 PDF

- [Federated Graph Anomaly Detection via Disentangled Representation Learning（OpenReview PDF）](https://openreview.net/pdf?id=xqjnhRqdK9)：网页端可访问，当前下载环境返回 403，因此没有留下不完整的本地 PDF；目前未找到已确认的作者仓库。该文研究联邦训练，不属于 training-free 方法。

## 综述与基础阅读

- [Survey-Brief-GAD-2024.pdf](surveys/Survey-Brief-GAD-2024.pdf)
- [Survey-Deep-GAD-Ma-TKDE-2023.pdf](surveys/Survey-Deep-GAD-Ma-TKDE-2023.pdf)
- [Survey-Deep-GAD-Qiao-TKDE-2025.pdf](surveys/Survey-Deep-GAD-Qiao-TKDE-2025.pdf)
- [Survey-Graph-Anomaly-Akoglu-2014.pdf](surveys/Survey-Graph-Anomaly-Akoglu-2014.pdf)

## 工具与相关资料

- [`PyGOD-JMLR-2024.pdf`](tools/PyGOD-JMLR-2024.pdf)：PyGOD 图异常检测工具库论文。
- [`NOT-DOMINANT-UIUC-Active-Graph-Anomaly-Thesis-2024.pdf`](related-work/NOT-DOMINANT-UIUC-Active-Graph-Anomaly-Thesis-2024.pdf)：UIUC 硕士论文，主题是 active graph anomaly detection，**不是 DOMINANT 原论文**。
- [`ASU-link-returns-HTML-not-PDF.html`](access-issues/ASU-link-returns-HTML-not-PDF.html)：旧 ASU 链接返回的 HTML 错误页，**不是论文 PDF**，仅保留作来源核查记录。
