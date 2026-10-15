# 论文与代码来源核对

原有来源核对：2026-10-09；新增推荐论文与代码仓库核对：2026-10-12

## TFGAD

- 论文 PDF：[OpenReview attachment](https://openreview.net/attachment?id=OG1qScEqHK&name=pdf)
- 代码：[chxngzhou/TFGAD](https://github.com/chxngzhou/TFGAD)
- 仓库自述为论文实现；README 指向 GADAM 获取更多数据。

## FreeGAD

- 论文 PDF：[arXiv](https://arxiv.org/pdf/2508.10594)
- 代码：[yunf-zhao/FreeGAD](https://github.com/yunf-zhao/FreeGAD)
- 仓库 README 明确称其为论文官方实现，并提供依赖及数据入口。

## CoLA

- 论文 PDF：[Griffith 机构仓储](https://research-repository.griffith.edu.au/bitstreams/2ea7b9f9-608a-4ebc-95d9-ffc4f2c67f3e/download)
- 代码：[TrustAGI-Lab/CoLA](https://github.com/TrustAGI-Lab/CoLA)
- 仓库 README 明确对应 TNNLS CoLA 论文。其 README 要求旧版 DGL，请单独建环境。

## DOMINANT

- 代码：[kaize0409/GCN_AnomalyDetection](https://github.com/kaize0409/GCN_AnomalyDetection)
- 论文出版页：[SIAM DOI 页面](https://doi.org/10.1137/1.9781611975673.67)
- [作者论文列表](https://kaize0409.github.io/Publications.html) 的 PDF 链接目前指向 SIAM 出版页；旧的 ASU 地址本次返回 HTML 错误页（非 PDF）。因此没有把错误响应留作论文文件。请先试试作者列表或 SIAM 页面是否能通过浏览器直接阅读。
- 代码仓库可先单独阅读，但严格复现仍需核对论文方法与实验设置。

## PyGOD（基线工具库）

- 论文：[JMLR 页面 / PDF](https://jmlr.org/papers/v25/23-0963.html)
- 代码：[pygod-team/pygod](https://github.com/pygod-team/pygod)
- PyGOD 包含 DOMINANT、CoLA 等多种检测器；它是库，不是单个模型。

## NeighborDiv

- 论文：本地 PDF `Papers/methods/NeighborDiv Training-free Zero-shot Generalist Graph Anomaly Detection via Neighbor Diversity，2026 年 arXiv 预印本.pdf`；arXiv 页面：<https://arxiv.org/abs/2605.20879>
- arXiv 标识：2605.20879v1，2026-05-20。
- 代码状态：本次检索未找到作者公开代码仓库；`Repositories/NeighborDiv/` 是根据论文第 3 节、实验设置和附录 B.4 编写的独立实现，不是作者代码的复现克隆。
- 数据情况：第一轮引用 `Repositories/CoLA/dataset/cora.mat`，该数据含 2,708 节点和 150 异常，与论文报告的 Cora 异常率相符，但当前文件边数与论文表 4 不一致，不能声称严格数据复现。细节见 [`NeighborDiv 复现记录`](../methods/NeighborDiv/NeighborDiv-复现记录与论文导读.md)。

## 2026-10-12 推荐论文与仓库对应

本节核对了近期推荐论文的全文、公开仓库及其在本项目中的位置。“未找到已确认的作者仓库”表示截至核对日的公开检索结果，不等同于证明作者从未发布代码。只下载/克隆资料，没有运行新增仓库的代码、测试或论文实验。

| 论文 | 可读全文 / 本地文件 | 对应代码仓库 | 项目状态 |
|---|---|---|---|
| **EB-GAD** — *Graph Anomaly Detection as Finite-Horizon Control: Training-Free Scoring via Empirical Bayes*（代码仓库使用 camera-ready 标题 *...as Dynamical Transport*） | [arXiv 全文](https://arxiv.org/html/2609.38424)；[`本地 PDF`](../../Papers/methods/EB-GAD-arXiv-2609.38424.pdf) | [heraclixus/EBGAD](https://github.com/heraclixus/EBGAD)；本地 `Repositories/EBGAD/` | 作者官方代码已公开；论文复现脚本在 `neurips2026/`。适合对照阅读，不能再归类为“缺官方代码”。 |
| **NeighborDiv** — *Training-free Zero-shot Generalist Graph Anomaly Detection via Neighbor Diversity* | [arXiv 全文](https://arxiv.org/html/2605.20879)；[`本地 PDF`](../../Papers/methods/NeighborDiv%20Training-free%20Zero-shot%20Generalist%20Graph%20Anomaly%20Detection%20via%20Neighbor%20Diversity%EF%BC%8C2026%20%E5%B9%B4%20arXiv%20%E9%A2%84%E5%8D%B0%E6%9C%AC.pdf) | 未找到已确认的作者仓库；本地 `Repositories/NeighborDiv/` | 本地目录是独立实现，不是作者代码克隆；已有复检/运行状态见 NeighborDiv 专项记录。 |
| **AG-Forge / TS-GGAD** — *Is Real-World Training Data Necessary for Generalist Graph Anomaly Detection?* | [arXiv 全文](https://arxiv.org/html/2610.12167)；[`本地 PDF`](../../Papers/methods/AG-Forge-TS-GGAD-arXiv-2610.12167.pdf) | 未找到已确认的作者仓库；未建立空仓库目录 | 可作为从论文实现的候选。论文包含合成图生成、训练数据构造和检测器，需要按论文逐部分实现并用论文表格核对。 |
| **Federated GAD** — *Federated Graph Anomaly Detection via Disentangled Representation Learning* | [OpenReview PDF](https://openreview.net/pdf?id=xqjnhRqdK9) | 未找到已确认的作者仓库 | 在线全文可访问；本机命令行下载返回 HTTP 403，故未保存本地 PDF。它研究联邦训练，与 training-free 主线不同。 |
| **SmoothGNN** — *Smoothing-aware GNN for Unsupervised Node Anomaly Detection* | [arXiv 预印本](https://arxiv.org/abs/2405.17525)；[`本地 PDF`](../../Papers/methods/SmoothGNN-arXiv-2405.17525.pdf) | [xydong127/SmoothGNN](https://github.com/xydong127/SmoothGNN)；本地 `Repositories/SmoothGNN/` | 作者公开实现；尚未开始本项目复现。 |
| **AlignGAD** — *A Zero-shot Generalized Graph Anomaly Detection Framework via Node Reconstruction* | [arXiv 全文](https://arxiv.org/abs/2606.12673)；[`本地 PDF`](../../Papers/methods/AlignGAD-arXiv-2606.12673.pdf) | [nhphan/AlignGAD](https://github.com/nhphan/AlignGAD)；本地 `Repositories/AlignGAD/` | 作者公开仓库已归档；尚未开始本项目复现。 |
| **ARC-zero** — *From Few-Shot to Zero-Shot: Towards Generalist Graph Anomaly Detection* | [arXiv 全文](https://arxiv.org/abs/2602.18793)；[`本地 PDF`](../../Papers/methods/ARC-zero-arXiv-2602.18793.pdf) | [Shiy-Li/ARC-zero](https://github.com/Shiy-Li/ARC-zero)；本地 `Repositories/ARC-zero/` | 作者官方 zero-shot 实现已归档；论文说明 few-shot ARC 代码另在 [Shiy-Li/ARC](https://github.com/Shiy-Li/ARC)。尚未开始本项目复现。 |

## 项目已有复现仓库对应

| 论文 / 方法 | 已归档仓库 |
|---|---|
| TFGAD — *Training-free Graph Anomaly Detection: A Simple Approach via Singular Value Decomposition* | [chxngzhou/TFGAD](https://github.com/chxngzhou/TFGAD)，本地 `Repositories/TFGAD/` |
| FreeGAD — *A Training-Free yet Effective Approach for Graph Anomaly Detection* | [yunf-zhao/FreeGAD](https://github.com/yunf-zhao/FreeGAD)，本地 `Repositories/FreeGAD/` |
| CoLA — *Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning* | [TrustAGI-Lab/CoLA](https://github.com/TrustAGI-Lab/CoLA)，本地 `Repositories/CoLA/` |
| DOMINANT — *Deep Anomaly Detection on Attributed Networks* | [作者代码库](https://github.com/kaize0409/GCN_AnomalyDetection)，本地 `Repositories/DOMINANT/`；另有 [PyTorch 版本](https://github.com/kaize0409/GCN_AnomalyDetection_pytorch) 位于 `Repositories/DOMINANT-PyTorch/` |
| PyGOD（工具库，非单篇模型论文） | [pygod-team/pygod](https://github.com/pygod-team/pygod)，本地 `Repositories/PyGOD/` |

## 下一步选题提示

若目标是“有完整论文与对照结果、但需要自己实现”，优先考虑 **AG-Forge / TS-GGAD**；目前没有确认的作者代码仓库。NeighborDiv 也有独立实现，但当前目录已存在，应先沿用其专项复检记录。EB-GAD、SmoothGNN、AlignGAD 和 ARC-zero 都能直接取得作者仓库，适合作为代码阅读或复现对照。

尚未入库且已有公开实现的近期候选，见 [`近期公开代码论文候选`](近期公开代码论文候选-2026-10-12.md)。其中明确区分 training-free、零样本迁移、少样本和测试时适配。
