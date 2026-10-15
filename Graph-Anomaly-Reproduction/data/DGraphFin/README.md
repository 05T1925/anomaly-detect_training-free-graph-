# DGraphFin 数据集获取状态

截至 2026-10-12，本目录尚未包含 DGraphFin 原始数据。EB-GAD 使用的标准 DGraphFin 版本约有 3,700,550 个节点、4,300,999 条边和 17 维特征；EB-GAD 还报告在约 1,225,601 个带标签节点上计算部分指标。

## 获取方式

1. 打开 [DGraphFin 官方数据页](https://dgraph.xinye.com/dataset)，按页面流程申请免费的数据访问账号。
2. 登录后下载官方 `DGraphFin.zip`（或页面当前提供的等价文件），把原始压缩包放在本目录。
3. 解压时保留官方原始文件名，并在 `data/manifest.csv` 登记来源、文件大小、SHA-256 和下载日期。
4. 运行任何实验前，先对照 [EB-GAD 官方数据说明](../../Repositories/EBGAD/DATA.md)和 `Repositories/EBGAD/fraud_data/DATASET_STATS.md` 核对标签映射、全图/标注子图及边方向处理。

下载需要用户本人完成官方注册。本仓库不存放账号或认证信息，也不尝试绕过访问控制。数据取得后可继续把副本整理到此目录，供不同实验复用。
