# Notes 学习与实验记录索引

笔记按“方法阅读”和“复现实验”归档。建议从论文 PDF 开始，结合对应代码阅读，再在实验记录中写下自己实际执行的命令与结果。

## 按方法学习

- **TFGAD**：[`首次复现与数据文件查看`](methods/TFGAD/TFGAD-首次复现与数据文件查看.md)。训练免除的 SVD 基线及首轮运行记录另见 [`TFGAD 实验日志`](experiments/reproduction-log.md)。
- **FreeGAD**：[`复现记录与论文导读`](methods/FreeGAD/FreeGAD-复现记录与论文导读.md)，同目录保留论文文本提取版，便于全文搜索。
- **CoLA**：[`复现记录与论文导读`](methods/CoLA/CoLA-复现记录与论文导读.md)和[`详细复现操作与代码逐段导读`](methods/CoLA/CoLA-详细复现操作与代码逐段导读.md)，同目录保留论文文本提取版。
- **EBGAD**：[首次复现与论文代码学习](methods/EBGAD/EBGAD-首次复现与论文代码学习.md)，实验日志见 [EBGAD 冒烟测试记录](experiments/EBGAD-reproduction-log.md)，目前尚未复现论文主表。
- **DOMINANT**：[`复现记录与论文导读`](methods/DOMINANT/DOMINANT-复现记录与论文导读.md)。注意区分 DOMINANT 原论文与相关硕士论文。
- **NeighborDiv**：[`复现记录与论文导读`](methods/NeighborDiv/NeighborDiv-复现记录与论文导读.md)、[`论文代码复检与后续验证`](methods/NeighborDiv/NeighborDiv-论文代码复检与后续验证.md)，同目录保留论文文本提取版；当前实现是按论文独立编写，新修改尚未运行测试。

## 实验与资料

- [`本机环境配置与运行指引`](本机环境配置与运行指引.md)：Windows CUDA 主环境、WSL 旧版环境和方法对应关系。
- [`reproduction-log.md`](experiments/reproduction-log.md)：TFGAD 实验日志。
- [`reproduction-log-template.md`](experiments/reproduction-log-template.md)：新增实验时可复制填写的模板。
- [`paper-and-code-sources.md`](references/paper-and-code-sources.md)：论文、代码和全文获取来源核验。
- [`近期公开代码论文候选`](references/近期公开代码论文候选-2026-10-12.md)：截至 2026-10-12 尚未入库、但已有公开实现的前沿 GAD 论文，按跨域、training-free 邻接、时序和文本图方向整理。
- [`paper-dataset-crosswalk.md`](references/paper-dataset-crosswalk.md)：近期论文—数据集—本地文件对应表、版本差异和尚待申请的数据。
- 推荐论文、仓库与本地文件的一一对应也见 [`论文—代码对应索引`](references/paper-and-code-sources.md)；该索引区分作者实现、独立实现和暂未找到代码的论文。
- `references/NOT-DOMINANT-UIUC-thesis-extracted.txt`：相关硕士论文的文本提取版，不是 DOMINANT 论文。

论文 PDF 目录按资料类别整理，详见 [`Papers/README.md`](../Papers/README.md)。各仓库的运行环境和数据路径以对应仓库说明及实验笔记为准。


