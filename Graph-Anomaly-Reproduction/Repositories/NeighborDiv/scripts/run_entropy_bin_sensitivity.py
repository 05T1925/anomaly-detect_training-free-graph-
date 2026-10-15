"""Explore Appendix E.1 entropy ablation under explicit, local bin choices.

The paper does not state the number or placement of entropy bins. This script
therefore records a sensitivity sweep rather than claiming an exact Table 11
reproduction. Sparse target graphs use Full; T-Finance uses the already studied
k=100 and k=50 sampling budgets, each with five pair seeds.
"""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_graph  # noqa: E402
from src.heuristic_baselines import calibrate_raw  # noqa: E402
from src.neighbordiv import _normalize_projected, _similarity_batches  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs/paper_experiments.json"
OUTPUT_DIR = WORKSPACE_ROOT / "results/NeighborDiv/paper-audit-entropy-bin-sensitivity"
BINS = (5, 10, 20, 50)
PAIR_BATCH_SIZE = 65536
RANK = 8
SVD_SEED = 0


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def entropy_counts(counts):
    total = int(counts.sum())
    if total == 0:
        return 0.0
    probabilities = counts[counts > 0].astype(np.float64) / float(total)
    return float(-np.dot(probabilities, np.log(probabilities)))


def projection(features):
    projector = TruncatedSVD(n_components=RANK, algorithm="randomized", n_iter=5,
                             random_state=SVD_SEED)
    projected = projector.fit_transform(features)
    return _normalize_projected(projected)[0]


def run_configuration(graph, projected, pair_seed, max_pairs):
    labels = graph.labels.astype(np.int64, copy=False)
    adjacency = graph.adjacency
    degree = np.diff(adjacency.indptr)
    valid = degree >= 2
    raw_by_bins = {n_bins: np.zeros(labels.size, dtype=np.float64) for n_bins in BINS}
    rng = np.random.default_rng(int(pair_seed))
    possible_pairs = evaluated_pairs = 0
    started = time.perf_counter()

    for node in np.flatnonzero(valid):
        neighbors = adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node + 1]]
        d = int(degree[node])
        possible = d * (d - 1) // 2
        possible_pairs += possible
        evaluated_pairs += possible if max_pairs is None else min(possible, max_pairs)
        counts_by_bins = {n_bins: np.zeros(n_bins, dtype=np.int64) for n_bins in BINS}
        for similarities in _similarity_batches(neighbors, projected, max_pairs,
                                               PAIR_BATCH_SIZE, rng):
            clipped = np.clip(similarities, -1.0, 1.0)
            for n_bins in BINS:
                counts_by_bins[n_bins] += np.histogram(clipped, bins=n_bins,
                                                       range=(-1.0, 1.0))[0]
        for n_bins in BINS:
            raw_by_bins[n_bins][node] = entropy_counts(counts_by_bins[n_bins])

    score_seconds = time.perf_counter() - started
    rows = []
    mode = "Full" if max_pairs is None else "k={}".format(max_pairs)
    for n_bins in BINS:
        scores = calibrate_raw(raw_by_bins[n_bins], valid)
        rows.append({
            "configuration": mode,
            "pair_seed": int(pair_seed),
            "bins": n_bins,
            "AUROC": float(roc_auc_score(labels, scores)),
            "AP": float(average_precision_score(labels, scores)),
            "n_nodes": int(labels.size),
            "n_anomalies": int(labels.sum()),
            "n_valid": int(valid.sum()),
            "possible_pairs": int(possible_pairs),
            "evaluated_pairs": int(evaluated_pairs),
            "scoring_seconds_shared_across_bins": float(score_seconds),
        })
    return rows


def main():
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--datasets', nargs='+', default=None)
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir.resolve()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for spec in config["datasets"]:
        if args.datasets and spec['name'] not in args.datasets:
            continue
        path = WORKSPACE_ROOT / "data" / spec["path"]
        print("loading {} from {}".format(spec["name"], path), flush=True)
        graph = load_graph(path, data_format=spec["format"], edge_policy=spec["edge_policy"],
                           load_labels=True)
        projected = projection(graph.features)
        if spec["name"] == "T-Finance":
            runs = [(k, seed) for k in (100, 50) for seed in range(5)]
        else:
            runs = [(None, 0)]
        for max_pairs, seed in runs:
            rows = run_configuration(graph, projected, seed, max_pairs)
            all_rows.extend([{ "dataset": spec["name"], **row} for row in rows])
            print("{} {} seed={}: {} bins, {:.3f}s".format(
                spec["name"], "Full" if max_pairs is None else "k={}".format(max_pairs),
                seed, len(BINS), rows[0]["scoring_seconds_shared_across_bins"]), flush=True)

    fields = ["dataset", "configuration", "pair_seed", "bins", "AUROC", "AP", "n_nodes",
              "n_anomalies", "n_valid", "possible_pairs", "evaluated_pairs",
              "scoring_seconds_shared_across_bins"]
    with (OUTPUT_DIR / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    aggregates = []
    keys = sorted({(r["dataset"], r["configuration"], r["bins"]) for r in all_rows})
    for dataset, configuration, n_bins in keys:
        values = [r for r in all_rows if (r["dataset"], r["configuration"], r["bins"]) ==
                  (dataset, configuration, n_bins)]
        aggregates.append({
            "dataset": dataset, "configuration": configuration, "bins": n_bins,
            "seeds": [r["pair_seed"] for r in values],
            "AUROC_mean": float(np.mean([r["AUROC"] for r in values])),
            "AUROC_std_population": float(np.std([r["AUROC"] for r in values], ddof=0)),
            "AP_mean": float(np.mean([r["AP"] for r in values])),
            "AP_std_population": float(np.std([r["AP"] for r in values], ddof=0)),
        })
    aggregate_fields = ["dataset", "configuration", "bins", "seeds", "AUROC_mean",
                        "AUROC_std_population", "AP_mean", "AP_std_population"]
    with (OUTPUT_DIR / "aggregate.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregates)

    metadata = {
        "paper_section": "Appendix E.1 entropy statistic sensitivity",
        "status": "local sensitivity only; exact Table 11 entropy result is under-specified",
        "entropy_bins": list(BINS),
        "bin_edges": "equal-width over cosine similarity [-1, 1]",
        "entropy_definition": "natural-log Shannon entropy over observed pair similarities",
        "valid_node_set": "degree>=2; low-degree scores use zero fallback and are excluded from calibration",
        "reference": "absolute deviation from the median across degree>=2 nodes; population z-score",
        "rank": RANK, "svd_random_state": SVD_SEED, "svd_algorithm": "sklearn randomized, n_iter=5",
        "non_tfinance_pair_mode": "Full",
        "tfinance_pair_mode": "k=100 and k=50, each with pair seeds 0..4; no Full entropy pass",
        "n_runs": len(all_rows), "metrics_csv": "metrics.csv", "aggregate_csv": "aggregate.csv",
        "data_sha256": {spec["name"]: sha256(WORKSPACE_ROOT / "data" / spec["path"])
                        for spec in config["datasets"]},
        "source_sha256": {
            "src/neighbordiv.py": sha256(PROJECT_ROOT / "src/neighbordiv.py"),
            "scripts/run_entropy_bin_sensitivity.py": sha256(Path(__file__).resolve()),
        },
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        },
        "notes": [
            "The paper does not publish entropy bin count, range, or bin-edge convention.",
            "T-Finance entropy values use sampling and are not directly comparable to the paper's unspecified Full Table 11 setting.",
        ],
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print("completed {} entropy configurations".format(len(all_rows)), flush=True)


if __name__ == "__main__":
    main()
