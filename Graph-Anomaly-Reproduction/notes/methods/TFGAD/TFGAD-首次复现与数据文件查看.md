# TFGAD 首次复现与数据文件查看

## 当前进度

已在 WSL Ubuntu 20.04 的独立 Python 虚拟环境中跑通仓库自带的 `inj_cora.bin`。代码仓库本身没有被修改。

环境版本：Python 3.8.10、PyTorch 2.1.0+cpu、DGL 1.1.3、NumPy 1.24.4、scikit-learn 1.3.2、SciPy 1.10.1。

验证输出：

```text
torch.Size([2708, 2708]) torch.Size([2708, 1433]) torch.Size([2708])
AUC-ROC: 0.9867, AUC-PR: 0.8180
```

截至 2026-10-09，Cora 已按论文参数运行五次；后续结果与五次统计见“后续运行记录”。运行时均未使用 `time`，所以 Cora 耗时尚未记录。

`inj_cora.bin` 内含 1 张图、2708 个节点、11604 条边；节点特征矩阵为 2708×1433，标签包含 2558 个正常节点和 150 个异常节点。

`torch.sparse.FloatTensor` 有弃用提醒，但此次运行成功。程序把 `average_precision_score` 打印为 `AUC-PR`；严格记实验时建议记作 AP/AUPRC，并在报告里说明采用的是 scikit-learn 的 Average Precision。第一次运行说明数据读取、图邻接矩阵转换、低秩分解和指标计算链路都通了；由于 `torch.svd_lowrank` 有随机性，后续多次运行的最后几位可能略有波动。

## 每次重新运行

在 Windows PowerShell 输入：

```powershell
wsl -d Ubuntu-20.04
```

进入 Ubuntu 终端后输入：

```bash
source ~/.venvs/tfgad/bin/activate
cd "/mnt/c/Users/28641/Desktop/论文科研-江权学长/Graph-Anomaly-Reproduction/Repositories/TFGAD"
export DGLBACKEND=pytorch
python main.py --dataset inj_cora --k1 1 --k2 5 --eta 10
```

数据集参数对应 `dataset/inj_cora.bin`。程序的 `dataset` 路径是相对当前目录的，所以运行前必须先 `cd` 到 TFGAD 仓库目录。

退出环境：

```bash
deactivate
```

## 环境是怎样配置的

本机的 Windows Python 环境缺少与该仓库 DGL 版本匹配的 Windows 安装包，因此复现使用已安装的 WSL Ubuntu 20.04。环境在 WSL 用户目录 `~/.venvs/tfgad`，不与其他 Python 项目混用。仓库脚本没有 GPU 调用，所以先装 PyTorch CPU 版即可；当前 `torch.cuda.is_available()` 为 `False`，这不妨碍此次小数据集实验。

若需要在另一台同类 WSL 环境重建，可按顺序运行：

```bash
python3 -m venv --help
```

Ubuntu 20.04 可能没预装 `python3-venv` 与 pip。安装它们需要系统管理员权限；当前机器已通过用户级 `virtualenv` 建好环境。环境建好后安装包：

```bash
~/.venvs/tfgad/bin/pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.1.0"
~/.venvs/tfgad/bin/pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple --find-links https://data.dgl.ai/wheels/repo.html "dgl==1.1.3" "numpy==1.24.4" "scikit-learn==1.3.2" "scipy==1.10.1" packaging
```

注意：TFGAD 仓库没有 `requirements.txt`。DGL 1.1.3 自带的稀疏扩展包含 PyTorch 2.1.0 对应的库，不包含 2.2.2 对应库；把 PyTorch 换成 2.2.2 会在 `g.adj()` 报 `libdgl_sparse_pytorch_2.2.2.so` 不存在。复现时保留上面这组版本。

## `.bin`、`.csv`、`.mat` 怎么打开

| 文件 | 是什么 | 建议方式 |
|---|---|---|
| `.bin` | 此仓库中是 DGL 序列化的图数据，不是普通文本或 Excel 文件 | 用 Python 和 `dgl.load_graphs` 读取；不要双击，也不要用文本编辑器直接打开 |
| `.csv` | 逗号分隔的表格 | Excel/WPS 可以查看；Python 可用 `pandas.read_csv` |
| `.mat` | MATLAB 数据文件，通常存矩阵、标签等变量 | Python 用 `scipy.io.loadmat`；较新的 MATLAB v7.3 文件改用 `h5py`；MATLAB/Octave 也能直接读 |

### 查看当前 `.bin` 里的图、特征和标签

在 WSL 里激活环境并进入 TFGAD 目录后运行：

```bash
python - <<'PY'
import dgl

graphs, metadata = dgl.load_graphs("dataset/inj_cora.bin")
g = graphs[0]
print("图数量:", len(graphs))
print("节点数:", g.num_nodes())
print("边数:", g.num_edges())
print("节点字段:", list(g.ndata.keys()))
print("特征形状:", tuple(g.ndata["feat"].shape))
print("标签形状:", tuple(g.ndata["label"].shape))
print("标签取值:", g.ndata["label"].unique(return_counts=True))
PY
```

TFGAD 当前程序需要每个图带有 `feat` 节点特征和 `label` 节点标签。

### 查看 `.csv`

```python
import pandas as pd

df = pd.read_csv("some_file.csv")
print(df.head())
print(df.shape)
print(df.dtypes)
```

### 查看 `.mat`

对普通 MAT 文件：

```python
from scipy.io import loadmat

data = loadmat("some_file.mat")
print([key for key in data if not key.startswith("__")])
for key, value in data.items():
    if not key.startswith("__"):
        print(key, getattr(value, "shape", type(value)))
```

如果提示 MATLAB v7.3/HDF5 格式，再改用：

```python
import h5py

with h5py.File("some_file.mat", "r") as f:
    print(list(f.keys()))
```

## 后续运行记录（2026-10-09）

### Cora：仓库默认参数 / 论文参数

五次运行使用同一命令 `python main.py --dataset inj_cora --k1 1 --k2 5 --eta 10`。论文 Table 7 的 Cora 参数也是 `k1=1`、`k2=5`、`eta=10`。

| 次数 | AUC-ROC | AP（程序打印为 AUC-PR） |
|---:|---:|---:|
| 1 | 0.9867 | 0.8180 |
| 2 | 0.9863 | 0.8174 |
| 3 | 0.9870 | 0.8218 |
| 4 | 0.9869 | 0.8214 |
| 5 | 0.9864 | 0.8179 |
| 五次均值 | 0.986660（约 0.9867） | 0.819300（约 0.8193） |
| 五次样本标准差 | 0.000305 | 0.002117 |

五次 ROC-AUC 均值四舍五入后与论文 Table 3 的 Cora 五次均值 `0.9867` 一致；AP 均值为 `0.8193`，比论文表中的 AUC-PR `0.8197` 低 `0.0004`。这是很接近的复现结果。代码将 `average_precision_score` 打印为 AUC-PR，论文表称 AUC-PR；若两边计算实现不同，数值不一定完全可比。这里的标准差由本次五个单次结果计算。五次运行都没有计时。

### Reddit：默认参数一次，论文参数六次

各次输出中的形状均为邻接矩阵 `10984×10984`、特征矩阵 `10984×64`、标签向量 `10984`。输出都显示稀疏张量弃用提醒，但程序正常完成。

| 参数组 | 命令 | AUC-ROC | AP（程序打印为 AUC-PR） | `real` 耗时 |
|---|---|---:|---:|---:|
| 仓库默认参数（1 次） | `--dataset inj_reddit --k1 1 --k2 5 --eta 10` | 0.5894 | 0.0417 | 未记录 |
| 论文 Reddit 参数（第 1 次） | `--dataset inj_reddit --k1 10 --k2 5 --eta 500` | 0.5988 | 0.0412 | 未记录 |
| 论文 Reddit 参数（第 2 次） | 同上 | 0.6009 | 0.0421 | 1.883 秒 |
| 论文 Reddit 参数（第 3 次） | 同上 | 0.6116 | 0.0439 | 18.311 秒 |
| 论文 Reddit 参数（第 4 次） | 同上 | 0.6011 | 0.0423 | 17.289 秒 |
| 论文 Reddit 参数（第 5 次） | 同上 | 0.5967 | 0.0415 | 16.257 秒 |
| 论文 Reddit 参数（第 6 次，额外） | 同上 | 0.6081 | 0.0428 | 12.346 秒 |
| 前五次均值 | 同上 | 0.601820 | 0.042200 | — |
| 前五次样本标准差 | 同上 | 0.005752 | 0.001049 | — |
| 六次均值（含额外一次） | 同上 | 0.602867 | 0.042300 | — |
| 论文 Table 3 五次均值 | 附录 Table 7 参数 | 0.6021 | 0.0423 | — |

论文参数第 2–6 次的完整 `real`、`user` 和 `sys` 计时已记入 `reproduction-log.md`。`real` 是墙钟时间，已记录值在 1.883–18.311 秒之间，波动较大；`user` 和 `sys` 是 CPU 时间，多线程下可能大于墙钟时间。论文效率实验排除了数据读取和模型初始化，不能直接把本机 `real` 与论文秒数严格比较。

按与论文相同的五次数量比较，前五次均值比论文 Table 3 低 `0.000280` ROC-AUC 和 `0.000100` AP；第六次作为额外结果也保留。默认参数与论文参数同时改变了 `k1` 和 `eta`，不能据此把结果变化归因到单独某个参数。

## 下一步复现顺序

1. Cora 已凑齐五次；五次均值 ROC-AUC `0.986660`、AP `0.819300`。若还要比较运行时间，可另行用 `time` 运行并注明这不是前五次的计时记录。
2. Reddit 论文参数已完成五次，并多做一次；前五次统计用于和论文五次结果对照，六次整体统计作为补充。默认参数结果单独保留。
3. 每次将日期、Git commit、环境版本、命令、输出数值和异常写入项目的 `reproduction-log-template.md` 副本。不要只记截图。
4. 指标对不上时，先核对数据文件版本、参数、随机性和论文报告的是均值还是单次结果，再查代码，不要一上来改算法实现。

论文附录 Table 6 给 Cora 的边数为 5,429，而当前 `inj_cora.bin` 经 DGL 读取有 11,604 条边。两者的计数口径或数据处理阶段可能不同，不能仅凭这两个数断定数据不一致；若要严格核对，需要再检查图中的有向边、反向边、自环及异常注入后的边，并确认论文表格统计口径。

## 代码阅读顺序

从 `main.py` 读起：`load_dataset` 负责读取图、特征、邻接矩阵和标签；`torch.svd_lowrank` 分别对特征矩阵与邻接矩阵做低秩分解；`att_rec_err` 与 `adj_prj_len` 形成节点异常分数；最后用标签计算 ROC-AUC 和 AP（代码打印为 AUC-PR）。这份仓库实现是 training-free：它没有训练循环、优化器或反向传播。
