"""Run Appendix D LCC/NRS/PCD/AMEN-Ego baselines on the seven target graphs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_graph  # noqa: E402
from src.heuristic_baselines import project_l1, score_baseline  # noqa: E402


DATASETS = ("Cora", "YelpChi", "Reddit", "T-Finance", "Tolokers", "Disney", "Questions")
METHODS = ("LCC", "NRS", "PCD", "AMEN-Ego")
OUT_DIR = REPO_ROOT / "results/NeighborDiv/paper-audit-appendix-d"
CONFIG_PATH = PROJECT_ROOT / "configs/paper_experiments.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    OUT_DIR = args.output_dir.resolve()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    specs = {entry["name"]: entry for entry in config["datasets"]}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_file = OUT_DIR / "metrics.csv"
    if metrics_file.is_file():
        with metrics_file.open(encoding="utf-8-sig", newline="") as stream:
            metrics = {(row["dataset"], row["method"]): row for row in csv.DictReader(stream)}
    else:
        metrics = {}
    metadata_file = OUT_DIR / "metadata.json"
    previous_metadata = json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.is_file() else {}
    datasets_recorded = list(dict.fromkeys(previous_metadata.get("datasets_requested", []) + args.datasets))
    methods_recorded = list(dict.fromkeys(previous_metadata.get("methods_requested", []) + args.methods))

    metadata = {
        "paper_section": "Appendix D, Table 12 heuristic baseline comparison",
        "datasets_requested": datasets_recorded,
        "methods_requested": methods_recorded,
        "rank": 8,
        "svd_random_state": 0,
        "svd_algorithm": "sklearn TruncatedSVD randomized; n_iter=5",
        "feature_preprocessing": "U Sigma, row-wise L1 normalization; AMEN-Ego then min-max rescales each dimension to [0,1]",
        "edge_policy": "undirected union for all baselines, as required by LCC and symmetric-normalized propagation",
        "calibration": "valid-set median, absolute deviation and population z-score; invalid nodes receive zero",
        "valid_node_sets": {"LCC": "degree>=2", "NRS": "degree>=1", "PCD": "degree>=1", "AMEN-Ego": "degree>=1"},
        "local_implementation_choices": [
            "The paper does not fully specify input edge symmetrization for every dataset; baselines use an undirected union consistently.",
            "AMEN-Ego applies the size-invariant AMEN internal/external rescaling and reports negative optimized normality as anomaly score.",
            "For graphs with an estimated AMEN ego-edge workload of at least 1e9, an OpenMP bitset kernel computes the same score; smaller graphs use the direct Python reference path.",
        ],
        "source_sha256": {
            "src/heuristic_baselines.py": sha256(PROJECT_ROOT / "src/heuristic_baselines.py"),
            "src/amen_ego_native.cpp": sha256(PROJECT_ROOT / "src/amen_ego_native.cpp"),
            "scripts/run_appendix_d_heuristics.py": sha256(Path(__file__).resolve()),
        },
    }
    metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for name in args.datasets:
        spec = specs[name]
        source = REPO_ROOT / "data" / spec["path"]
        print("loading {} from {}".format(name, source), flush=True)
        loaded_at = time.perf_counter()
        graph = load_graph(source, data_format=spec["format"], edge_policy=spec["edge_policy"], load_labels=True)
        labels = graph.labels.astype(np.int64, copy=False)
        data_hash = sha256(source)
        loaded_seconds = time.perf_counter() - loaded_at
        features = project_l1(graph.features, rank=8, random_state=0, n_iter=5)
        folder = OUT_DIR / "scores" / name
        folder.mkdir(parents=True, exist_ok=True)
        label_folder = OUT_DIR / "labels"
        label_folder.mkdir(parents=True, exist_ok=True)
        with (label_folder / (name.replace("-", "_") + ".csv")).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["node_id", "label"])
            writer.writerows(enumerate(labels.tolist()))
        print("{} loaded: nodes={}, features={}, labels={}, load_seconds={:.3f}".format(
            name, graph.adjacency.shape[0], features.shape[1], int(labels.sum()), loaded_seconds), flush=True)
        for method in args.methods:
            started = time.perf_counter()
            scores, raw, valid = score_baseline(method, graph.adjacency, features)
            if not np.isfinite(scores).all() or scores.shape != labels.shape:
                raise ValueError("Invalid calibrated score vector for {} / {}".format(name, method))
            metrics[(name, method)] = {
                "dataset": name,
                "method": method,
                "AUROC": "" if np.unique(labels).size < 2 else "{:.17g}".format(roc_auc_score(labels, scores)),
                "AP": "" if np.unique(labels).size < 2 else "{:.17g}".format(average_precision_score(labels, scores)),
                "n_nodes": int(labels.size),
                "n_anomalies": int(labels.sum()),
                "n_valid": int(valid.sum()),
                "n_edges_undirected": int(graph.adjacency.maximum(graph.adjacency.T).nnz // 2),
                "scoring_seconds": "{:.9f}".format(time.perf_counter() - started),
                "data_sha256": data_hash,
            }
            score_file = folder / (method.replace("-", "_") + ".csv")
            with score_file.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["node_id", "raw_score", "calibrated_score", "valid"])
                writer.writerows((i, "{:.17g}".format(raw[i]), "{:.17g}".format(scores[i]), int(valid[i]))
                                 for i in range(labels.size))
            print("{} {}: AUROC={} AP={} scoring_seconds={}".format(
                name, method, metrics[(name, method)]["AUROC"], metrics[(name, method)]["AP"],
                metrics[(name, method)]["scoring_seconds"]), flush=True)

            with metrics_file.open("w", newline="", encoding="utf-8") as stream:
                fieldnames = ["dataset", "method", "AUROC", "AP", "n_nodes", "n_anomalies", "n_valid",
                              "n_edges_undirected", "scoring_seconds", "data_sha256"]
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                for key in sorted(metrics):
                    writer.writerow(metrics[key])


if __name__ == "__main__":
    main()
