# ARC<sub>zero</sub>: Towards Generalist Graph Anomaly Detection

Official implementation of **ARC<sub>zero</sub>**, the zero-shot extension of ARC proposed in our paper *"From Few-Shot to Zero-Shot: Towards Generalist Graph Anomaly Detection"*. ARC<sub>zero</sub> is a single unified model that, once trained on a small set of source graphs, can directly detect anomalies on new unseen graphs **without any labeled node and without any dataset-specific fine-tuning**.

> The few-shot generalist version (ARC) is released separately at [Shiy-Li/ARC](https://github.com/Shiy-Li/ARC); we inherit the environment setup, dataset preprocessing pipeline, and cross-dataset evaluation protocol from that repository.

## Repository Structure

```
ARECZero-Github/
├── main.py                 # Training & evaluation entrypoint
├── model.py                # ARCZero backbone and cross-attention scorer
├── data.py                 # Dataset loading, feature alignment, propagation, shot selection
├── trainer.py              # Training loop and iterative inference
├── utils.py                # Sparse utils, metrics, JSON config loader
├── best_param_pt/          # Recommended hyperparameter JSONs for different shot numbers
└── dataset/                # Place the raw .mat graph datasets here
```

## Requirements

The environment is identical to [ARC](https://github.com/Shiy-Li/ARC).

- `torch`
- `torch-geometric`
- `numpy`
- `scipy`
- `scikit-learn`

Please refer to the [ARC repository](https://github.com/Shiy-Li/ARC) for the full `requirements.txt` and for CUDA / `torch-geometric` compatibility notes.

## Datasets

We follow exactly the same cross-dataset benchmark used in [ARC](https://github.com/Shiy-Li/ARC). For the dataset sources, please see the [ARC repository](https://github.com/Shiy-Li/ARC); ARC<sub>zero</sub> uses the same data splits and preprocessing protocol.

## Quick Start

From the repository root:

```bash
python main.py --device cuda:0 --trials 10 --random_shot 10
```
