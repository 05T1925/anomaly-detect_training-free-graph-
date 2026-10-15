"""Single graph experiment entry point. Only main()/run_experiment execute it."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_graph  # noqa: E402
from src.neighbordiv import evaluate_scores, predict_anomalies, score_graph  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser():
    parser = argparse.ArgumentParser(description="Run the from-paper NeighborDiv reproduction.")
    parser.add_argument("--data", required=True, help="Path to a local graph file.")
    parser.add_argument("--format", choices=("auto", "mat", "npz", "dgl", "dgl-zip"), default="auto")
    parser.add_argument("--adjacency-key", default=None)
    parser.add_argument("--feature-key", default=None)
    parser.add_argument("--label-key", default=None)
    parser.add_argument("--no-labels", action="store_true", help="Do not load labels; skip evaluation.")
    parser.add_argument("--edge-policy", choices=("auto", "preserve", "incoming", "undirected"), default="auto",
                        help="auto: undirected for Yandex NPZ; preserve for MAT/DGL.")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=None, help="Unset = Full, 100/50 = sampling ablation.")
    parser.add_argument("--pair-batch-size", type=int, default=65536, help="Memory chunk size; does not cap full pairs.")
    parser.add_argument("--full-backend", choices=("stream", "moments"), default="stream",
                        help="moments computes exact all-pair moments algebraically; stream enumerates every pair.")
    parser.add_argument("--random-state", type=int, default=0, help="Legacy seed used unless a separate seed is supplied.")
    parser.add_argument("--svd-random-state", type=int, default=None)
    parser.add_argument("--pair-random-state", type=int, default=None)
    parser.add_argument("--svd-algorithm", choices=("randomized", "arpack"), default="randomized")
    parser.add_argument("--svd-n-iter", type=int, default=5)
    parser.add_argument("--diversity-statistic", choices=("variance", "std", "mean"), default="variance")
    parser.add_argument("--reference", choices=("median", "mean"), default="median")
    parser.add_argument("--low-degree-fallback", choices=("zero", "median"), default="zero")
    parser.add_argument("--evaluation-scope", choices=("all", "valid"), default="all")
    parser.add_argument("--threshold-lambda", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly allow replacement of run output files.")
    return parser


def run_experiment(args):
    data_path = Path(args.data).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError("Data file does not exist: {}".format(data_path))
    if out_dir.exists() and (not out_dir.is_dir() or (any(out_dir.iterdir()) and not args.overwrite)):
        raise FileExistsError("Output must be a new/empty directory; choose another --output-dir or use --overwrite.")
    if not np.isfinite(args.threshold_lambda) or args.threshold_lambda < 0:
        raise ValueError("--threshold-lambda must be finite and nonnegative.")

    started = time.perf_counter()
    graph = load_graph(data_path, data_format=args.format, adjacency_key=args.adjacency_key,
                       feature_key=args.feature_key, label_key=args.label_key,
                       edge_policy=args.edge_policy, load_labels=not args.no_labels)
    loaded = time.perf_counter()
    result = score_graph(
        graph.adjacency, graph.features, rank=args.rank, max_pairs=args.max_pairs,
        random_state=args.random_state, svd_random_state=args.svd_random_state,
        pair_random_state=args.pair_random_state, svd_algorithm=args.svd_algorithm,
        svd_n_iter=args.svd_n_iter, pair_batch_size=args.pair_batch_size,
        diversity_statistic=args.diversity_statistic, reference=args.reference,
        low_degree_fallback=args.low_degree_fallback, full_backend=args.full_backend)
    scored = time.perf_counter()
    predictions, threshold_info = predict_anomalies(result.scores, args.threshold_lambda)
    mask = result.valid if args.evaluation_scope == "valid" else None
    metrics = evaluate_scores(graph.labels, result.scores, mask)
    evaluated = time.perf_counter()
    metadata = result.metadata
    parameters = {key: getattr(args, key) for key in (
        "rank", "max_pairs", "random_state", "svd_random_state", "pair_random_state",
        "svd_algorithm", "svd_n_iter", "pair_batch_size", "diversity_statistic",
        "reference", "low_degree_fallback", "evaluation_scope", "threshold_lambda", "edge_policy", "full_backend")}
    run = {
        "schema_version": 2,
        "paper": "NeighborDiv: Training-free Zero-shot Generalist Graph Anomaly Detection via Neighbor Diversity",
        "arxiv": "2605.20879v1", "data_path": str(data_path), "data_sha256": sha256(data_path),
        "data_format": graph.input_info["format"], "mat_keys": graph.keys, "input": graph.input_info,
        "parameters": parameters, "graph": metadata, "metrics": metrics,
        "binary_decision": dict(threshold_info, predicted_anomalies=int(predictions.sum())),
        "load_seconds": loaded - started, "scoring_seconds": scored - loaded,
        "evaluation_seconds": evaluated - scored,
        "load_and_scoring_seconds": scored - started,
        "end_to_end_compute_seconds": evaluated - started,
        "timing_scope": "CPU wall time; hash and file writes excluded; DGL ZIP load includes temporary extraction",
        "label_use": "evaluation only; never passed to score_graph, threshold, SVD or pair sampling",
        "implementation": "from-paper NumPy/SciPy/scikit-learn; no gradient training; not author code",
        "environment": {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
                        "processor": platform.processor(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "scikit_learn": sklearn.__version__},
        "source_sha256": {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in (
            PROJECT_ROOT / "src/data.py", PROJECT_ROOT / "src/neighbordiv.py", Path(__file__).resolve())},
        "zero_denominator_policy": "all valid scores 0 if deviation std == 0; graph flag records degeneration",
        "verification": "this file is generated only by an executed run; compare with the paper separately",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "anomaly_scores.csv",
               np.column_stack((np.arange(result.scores.size), result.scores)), delimiter=",",
               header="node_id,anomaly_score", comments="", fmt=["%d", "%.17g"])
    # Undefined diversity/deviation for degree<2 is explicitly NaN, not fake zero diversity.
    np.savetxt(out_dir / "node_diagnostics.csv",
               np.column_stack((np.arange(result.scores.size), result.degree, result.valid.astype(int),
                                result.diversity, result.deviation, result.scores, predictions)),
               delimiter=",", header="node_id,degree,valid,diversity,deviation,anomaly_score,predicted_anomaly",
               comments="", fmt=["%d", "%d", "%d", "%.17g", "%.17g", "%.17g", "%d"])
    (out_dir / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                     encoding="utf-8")
    print("data: {}".format(data_path))
    print("nodes: {}; adjacency entries: {}; valid nodes: {}".format(
        metadata["n_nodes"], metadata["n_edges_directed"], metadata["n_valid_nodes_degree_ge_2"]))
    print("rank={}; pair_mode={}; edge_policy={}".format(args.rank, metadata["pair_mode"], graph.input_info["edge_policy"]))
    print("AUROC={}; AP={}; scoring_seconds={:.3f}".format(metrics["AUROC"], metrics["AP"], scored - loaded))
    print("outputs: {}".format(out_dir))
    return run


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_experiment(args)
    except (ValueError, KeyError, FileNotFoundError, FileExistsError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
