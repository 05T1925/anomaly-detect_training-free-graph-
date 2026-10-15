# FreeGAD 复现记录与论文导读

## 复现状态

已在本机 WSL Ubuntu 20.04 中复用 `~/.venvs/tfgad` 环境运行仓库代码；未修改 FreeGAD 源码。实际代码位于仓库内层目录：

```text
Graph-Anomaly-Reproduction/Repositories/FreeGAD/FreeGAD/
```

环境：Python 3.8.10、PyTorch 2.1.0+cpu、NumPy 1.24.4、SciPy 1.10.1、scikit-learn 1.3.2。`run.py` 实际调用到的代码没有导入 DGL 或 PyTorch Geometric，因此当前这次运行不需要 README 中的 DGL、torch-scatter、torch-sparse、CUDA 等安装项。README 中给出的完整环境偏向作者的 GPU 环境；本次 CPU 运行验证了仓库中的 FreeGAD 主流程。

仓库当前短 commit：`c1b007d`。

### 本地数据结果与论文 Table 2 对比

指标统一换算成百分比。表中“本地”是固定 seed=0 对每个数据集执行一次；FreeGAD 没有随机初始化或训练阶段，同一数据多次运行应给出相同分数。

| 数据集 | 本地 AUROC | 论文 AUROC | 本地 AP/AUPRC | 论文 AUPRC | 核对 |
|---|---:|---:|---:|---:|---|
| Cora | 84.85 | 84.85 | 49.77 | 49.75 | 对上，差异为显示精度范围内 |
| Reddit | 57.21 | 57.21 | 3.85 | 3.85 | 对上 |
| YelpChi | 78.55 | 78.55 | 15.80 | 15.80 | 对上 |
| Flickr | 74.69 | 74.73 | 38.57 | 38.68 | 很接近，差约 0.04/0.11 个百分点 |
| Amazon | 63.41 | 88.57 | 8.48 | 75.06 | 不一致，需先核对数据版本 |

Amazon 文件的矩阵形状是 10,224×10,224，论文 Table 1 写的是 10,244 个节点；本地文件有 693 个正类节点、351,216 个非零邻接项。该文件与论文实验所用数据至少在节点数上不一致，当前 Amazon 结果不能作为同版本复现结论。不要为了追分数手工改代码或标签，先找到作者发布的同版本 `Amazon.mat` 再比较。

复现来源：本地 PDF [FreeGAD-CIKM-2025.pdf](../../../Papers/methods/FreeGAD-CIKM-2025.pdf)，Table 1/2/5；代码仓库 README 引用 arXiv:2508.10594。PDF 文本提取副本在同目录的 `FreeGAD-paper-extracted.txt`，方便搜索公式和表格。

## 怎样运行单个数据集

先在 PowerShell 进入 WSL：

```powershell
wsl -d Ubuntu-20.04
```

再在 Ubuntu 里激活环境，并进入**内层**代码目录：

```bash
source ~/.venvs/tfgad/bin/activate
cd "/mnt/c/Users/28641/Desktop/论文科研-江权学长/Graph-Anomaly-Reproduction/Repositories/FreeGAD/FreeGAD"
```

仓库当前的 `run.py` 会忽略 `--dataset` 参数，并按顺序循环 10 个数据集。你本地只有 Amazon、Reddit、YelpChi、Cora、Flickr 五个 MAT 文件，所以直接执行 `python run.py --dataset cora` 仍会跑整批任务，并在缺失的 Tolokers 等数据集处报错。

若只复现 Cora，用下面的命令调用仓库自身的参数读取和模型代码，不改源文件：

```bash
python - <<'PY'
from types import SimpleNamespace
from run import run_csv_param
from utils import set_seed

args = SimpleNamespace(device="cpu", model="FreeGAD")
set_seed(0)
auc, ap = run_csv_param("./params/FreeGAD.csv", args, "cora")
print(f"Cora AUROC={auc:.6f} ({auc*100:.2f}%), AP={ap:.6f} ({ap*100:.2f}%)")
PY
```

已验证输出约为 `AUROC=0.848486 (84.85%)`、`AP=0.497654 (49.77%)`。论文报告 `AUROC=84.85%`、`AUPRC=49.75%`。

### 一次跑完本地已有的 5 个数据集

```bash
python - <<'PY'
from pathlib import Path
from types import SimpleNamespace
from run import run_csv_param
from utils import set_seed

for dataset in ["Amazon", "Reddit", "YelpChi", "cora", "Flickr"]:
    if not Path(f"dataset/{dataset}.mat").exists():
        print(dataset, "缺少数据，跳过")
        continue
    args = SimpleNamespace(device="cpu", model="FreeGAD")
    set_seed(0)
    auc, ap = run_csv_param("./params/FreeGAD.csv", args, dataset)
    print(f"{dataset}: AUROC={auc:.4%}, AP={ap:.4%}")
PY
```

## 参数从哪里来

默认参数不是 `run.py` 开头写的 CLI 默认值，而是 `params/FreeGAD.csv` 中每个数据集对应的行；`run_csv_param()` 会读取该 CSV 并覆盖参数。例如 Cora 是 `alpha=0.010974988, beta=0.205651231, num_hops=7, num_shot=70`。仓库文档之外的数据集需要先取得对应 MAT 文件，CSV 里有参数并不代表数据已经下载。

## 论文和代码怎么读

建议按下面顺序阅读 PDF：

1. **Abstract 和 Introduction**：研究动机是 GNN 异常检测训练成本高、全图训练难扩展；作者提出检验去掉训练后能否保留性能。
2. **Section 4.1 Affinity-Gated Residual Encoder**：看多跳传播与 affinity 门控，理解它如何在原始特征空间中混合各跳表示。
3. **Section 4.2 Anchor Node Selection**：看最高/最低 affinity 节点如何作为正、负锚点。这里正锚点是伪正常，负锚点是伪异常，不是人工标注。
4. **Section 4.3 Anchor-Guided Anomaly Scoring**：看节点到两类锚点的距离统计如何合成异常分数。
5. **Section 5.1 与 Table 1/2**：核数据集统计、指标和主结果；再看 **Table 3** 运行时间、**Table 5** 去模块消融。

对应代码：

| 代码位置 | 在做什么 |
|---|---|
| `utils.py::load_anomaly_detection_dataset` | 读 MAT 中 `Network`、`Attributes`、`Label`，按数据集做特征和邻接预处理 |
| `utils.py::propagated` | 反复计算稀疏邻接矩阵与节点特征的乘积，产生 0 到 L 跳表示 |
| `utils.py::Affinity_Gated_Residual_Encoder` | 用多跳特征和原始特征的余弦相似度做门控残差融合，再平均得到表示 |
| `model.py::FreeGAD` | 选最高/最低 affinity 的锚点并调用评分模块，最后计算 AUROC 与 AP |
| `model.py::Anchor_Guided_Anomaly_Scoring` | 汇总目标节点到正、负锚点的平均、最近、最远距离并形成分数 |
| `params/FreeGAD.csv` | 每个数据集的 `alpha`、`beta`、传播跳数和锚点数 |
| `run.py` | 多数据集、五个 seed 的运行循环及均值/标准差打印 |

读公式时抓住三件事即可：

- **多跳传播**：特征沿归一化邻接矩阵逐跳传播，保存每个 hop 的表示。
- **亲和力门控残差**：每一跳按照该节点与原始特征的余弦相似度，混合传播特征和原始特征；随后汇总多跳表示。
- **锚点评分**：与高亲和节点的距离代表“偏离正常”的证据，与低亲和节点的距离代表“像异常”的证据；`alpha`、`beta` 控制两侧距离项的权重，目标是让异常节点得分更高。

“training-free”表示没有可学习权重、训练 epoch、反向传播或优化器；不等于完全没有超参数。代码仍使用论文给定/实验选择的 `alpha`、`beta`、`num_hops`、`num_shot`。评估时仍要用 `Label` 计算指标，但标签没有参与锚点选择或得分计算。

## 下一步

1. 把 Cora 的命令再跑一次，确认结果稳定；目前连续 5 个 seed 已得到完全相同的 AUROC/AP。
2. 阅读上表列出的函数，按“数据读取 → 传播 → 门控 → 选锚点 → 打分 → 评估”顺序画一张流程图。
3. 将 Amazon 数据版本问题记入实验日志；后续查作者数据盘或论文引用的数据来源，确认 10,244 节点版本后再重跑。
4. 记录每次的仓库 commit、MAT 文件、参数 CSV 行、运行命令和 AUROC/AP。不要只记录最终分数。
