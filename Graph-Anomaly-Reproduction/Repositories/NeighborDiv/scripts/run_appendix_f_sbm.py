"""Run the controlled-homophily SBM sweep described in Appendix F."""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.heuristic_baselines import project_l1, score_baseline  # noqa: E402
from src.neighbordiv import score_graph  # noqa: E402


OUT_DIR = REPO_ROOT / "results/NeighborDiv/paper-audit-appendix-f/sbm"
N, K, AVG_DEGREE, FEATURE_DIM = 2000, 5, 15, 50
HOMOPHILIES = (0.1, 0.3, 0.5, 0.7, 0.9)
SEEDS = (0, 1, 2)
ANOMALY_TYPES = ("Type-H", "Type-D", "Mixed")
BASELINES = ("NRS", "AMEN-Ego")


def make_normal_sbm(seed: int, h: float):
    rng = np.random.default_rng(seed)
    group_size = N // K
    groups = np.repeat(np.arange(K, dtype=np.int64), group_size)
    p_in = h * AVG_DEGREE / (group_size - 1)
    p_out = (1.0 - h) * AVG_DEGREE / (N - group_size)
    row_parts, col_parts = [], []
    for left_group in range(K):
        left_nodes = np.flatnonzero(groups == left_group)
        rr, cc = np.triu_indices(left_nodes.size, k=1)
        keep = rng.random(rr.size) < p_in
        row_parts.append(left_nodes[rr[keep]])
        col_parts.append(left_nodes[cc[keep]])
        for right_group in range(left_group + 1, K):
            right_nodes = np.flatnonzero(groups == right_group)
            keep = rng.random((left_nodes.size, right_nodes.size)) < p_out
            local_r, local_c = np.nonzero(keep)
            row_parts.append(left_nodes[local_r])
            col_parts.append(right_nodes[local_c])
    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts)
    adjacency = sp.coo_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(N, N)).tocsr()
    adjacency = (adjacency + adjacency.T).astype(bool).tocsr()
    adjacency.eliminate_zeros()

    centers = rng.normal(loc=0.0, scale=3.0, size=(K, FEATURE_DIM))
    features = centers[groups] + rng.normal(loc=0.0, scale=1.0, size=(N, FEATURE_DIM))
    degrees = np.diff(adjacency.indptr).astype(np.int64)
    return adjacency, sp.csr_matrix(features), groups, degrees, float(p_in), float(p_out)


def evaluate_run(h, seed, anomaly_type, adjacency, features, labels, groups, p_in, p_out):
    nd = score_graph(adjacency, features, rank=8, max_pairs=None,
                     svd_random_state=0, pair_random_state=0, svd_algorithm="randomized", svd_n_iter=5)
    projected = project_l1(features, rank=8, random_state=0, n_iter=5)
    scores = {"NeighborDiv": nd.scores}
    for baseline in BASELINES:
        scores[baseline] = score_baseline(baseline, adjacency, projected)[0]
    rows = []
    for method, prediction in scores.items():
        rows.append({
            "homophily": h, "generation_seed": seed, "anomaly_type": anomaly_type,
            "method": method,
            "AUROC": float(roc_auc_score(labels, prediction)),
            "AP": float(average_precision_score(labels, prediction)),
            "n_anomalies": int(labels.sum()),
            "mean_degree": float(np.diff(adjacency.indptr).mean()),
            "valid_nodes_degree_ge_1": int((np.diff(adjacency.indptr) >= 1).sum()),
            "valid_nodes_degree_ge_2": int((np.diff(adjacency.indptr) >= 2).sum()),
            "p_in": p_in, "p_out": p_out,
        })
    coo = adjacency.tocoo()
    deg = np.diff(adjacency.indptr)
    own = np.zeros(N)
    np.add.at(own, coo.row, (groups[coo.row] == groups[coo.col]).astype(float))
    ratios = np.divide(own, deg, out=np.zeros(N), where=deg > 0)
    anomaly = labels == 1
    audit = {"homophily_target": h, "seed_index": seed, "anomaly_type": anomaly_type,
             "realized_edge_homophily": float((groups[coo.row] == groups[coo.col]).mean()),
             "mean_normal_degree": float(deg[~anomaly].mean()),
             "mean_anomaly_degree": float(deg[anomaly].mean()),
             "mean_normal_own_community_fraction": float(ratios[~anomaly].mean()),
             "mean_anomaly_own_community_fraction": float(ratios[anomaly].mean()),
             "median_normal_pair_variance": float(np.median(nd.diversity[nd.valid & ~anomaly])),
             "median_anomaly_pair_variance": float(np.median(nd.diversity[nd.valid & anomaly])),
             "background_seed": int(seed * 100 + round(h * 10)),
             "anomaly_selection_seed": int(seed * 10000 + round(h * 100)),
             "rewire_seed_H": int(seed * 100000 + round(h * 1000) * 10 + ANOMALY_TYPES.index(anomaly_type) * 2) if anomaly_type != 'Type-D' else None,
             "rewire_seed_D": int(seed * 100000 + round(h * 1000) * 10 + ANOMALY_TYPES.index(anomaly_type) * 2 + 1) if anomaly_type != 'Type-H' else None}
    return rows, audit


def main():
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    OUT_DIR = parser.parse_args().output_dir.resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    graph_audits = []
    completed_graphs = 0
    total_graphs = len(SEEDS) * len(HOMOPHILIES) * len(ANOMALY_TYPES)
    for seed in SEEDS:
        for h in HOMOPHILIES:
            combined_seed = int(seed * 100 + round(h * 10))
            base_adj, features, groups, base_degree, p_in, p_out = make_normal_sbm(combined_seed, h)
            anomaly_rng = np.random.default_rng(seed * 10000 + round(h * 100))
            selected = anomaly_rng.choice(N, size=100, replace=False)
            h_nodes, d_nodes = selected[:50], selected[50:]
            anomaly_sets = {
                "Type-H": h_nodes,
                "Type-D": d_nodes,
                "Mixed": selected,
            }
            for type_index, anomaly_type in enumerate(ANOMALY_TYPES):
                anomaly_nodes = anomaly_sets[anomaly_type]
                type_h = anomaly_type in ("Type-H", "Mixed")
                type_d = anomaly_type in ("Type-D", "Mixed")
                pieces = []
                labels = np.zeros(N, dtype=np.int64)
                if type_h:
                    pieces.append((h_nodes, False, 0))
                    labels[h_nodes] = 1
                if type_d:
                    pieces.append((d_nodes, True, 1))
                    labels[d_nodes] = 1
                adjacency = base_adj
                current_mask = np.zeros(N, dtype=bool)
                current_mask[anomaly_nodes] = True
                coo = base_adj.tocoo()
                keep = ~(current_mask[coo.row] | current_mask[coo.col])
                rows = [coo.row[keep].tolist()]
                cols = [coo.col[keep].tolist()]
                for nodes, diverse, part in pieces:
                    rng = np.random.default_rng(seed * 100000 + round(h * 1000) * 10 + type_index * 2 + part)
                    normal_nodes = np.flatnonzero(~current_mask)
                    for node in nodes:
                        if diverse:
                            candidates = normal_nodes[normal_nodes != node]
                        else:
                            candidates = np.flatnonzero((groups == groups[node]) & ~current_mask)
                        count = min(int(base_degree[node]), candidates.size)
                        if count:
                            targets = rng.choice(candidates, size=count, replace=False)
                            rows[0].extend([int(node)] * count)
                            cols[0].extend(targets.tolist())
                            rows[0].extend(targets.tolist())
                            cols[0].extend([int(node)] * count)
                adjacency = sp.coo_matrix((np.ones(len(rows[0]), dtype=np.int8),
                                           (np.asarray(rows[0]), np.asarray(cols[0]))), shape=(N, N)).tocsr()
                adjacency.data[:] = 1
                adjacency.setdiag(0)
                adjacency.eliminate_zeros()
                rows_result, graph_audit = evaluate_run(h, seed, anomaly_type, adjacency, features, labels,
                                                       groups, p_in, p_out)
                results.extend(rows_result)
                graph_audits.append(graph_audit)
                completed_graphs += 1
                print("completed {}/{}: h={}, seed={}, type={}".format(
                    completed_graphs, total_graphs, h, seed, anomaly_type), flush=True)

    fieldnames = ["homophily", "generation_seed", "anomaly_type", "method", "AUROC", "AP",
                  "n_anomalies", "mean_degree", "valid_nodes_degree_ge_1", "valid_nodes_degree_ge_2",
                  "p_in", "p_out"]
    with (OUT_DIR / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    table = []
    for anomaly_type in ANOMALY_TYPES:
        for method in ("NeighborDiv",) + BASELINES:
            for h in HOMOPHILIES:
                rows = [r for r in results if r["anomaly_type"] == anomaly_type and r["method"] == method and r["homophily"] == h]
                for metric in ("AUROC", "AP"):
                    values = np.asarray([r[metric] for r in rows], dtype=np.float64)
                    table.append({"anomaly_type": anomaly_type, "method": method, "homophily": h,
                                  "metric": metric, "mean": float(values.mean()),
                                  "std_population": float(values.std(ddof=0)), "n_seeds": int(values.size)})
    with (OUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["anomaly_type", "method", "homophily", "metric", "mean", "std_population", "n_seeds"])
        writer.writeheader()
        writer.writerows(table)

    metadata = {
        "paper_section": "Appendix F, controlled homophily sweep",
        "n": N, "communities": K, "community_sizes": [N // K] * K,
        "expected_average_degree": AVG_DEGREE, "feature_dimensions": FEATURE_DIM,
        "homophily_values": list(HOMOPHILIES), "generation_seeds": list(SEEDS),
        "anomaly_types": list(ANOMALY_TYPES), "anomalies_per_type": 50,
        "svd_random_state": 0, "neighbor_div_pair_mode": "Full",
        "local_protocol_choices": [
            "target p_in and p_out allocate expected degree 15 in proportions h and 1-h",
            "each injected node keeps its sampled background degree after rewiring",
            "rewired endpoints are sampled without replacement from normal nodes; Type-D samples uniformly over nodes, which gives equal-sized communities equal probability",
            "seed indices are 0,1,2; actual background/selection/rewiring seed formulas and realized graph statistics are in graph_audit.json",
        ],
        "reference_table": "Table 18 reports AUC only; this run also records AP as an additional local metric",
        "runs_csv": "runs.csv", "summary_csv": "summary.csv",
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT_DIR / "graph_audit.json").write_text(json.dumps(graph_audits, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "source_sha256.json").write_text(json.dumps({str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__).resolve(), PROJECT_ROOT/'src/neighbordiv.py', PROJECT_ROOT/'src/heuristic_baselines.py')},indent=2))
    print("completed runs: {}".format(len(results) // 3))
    print("outputs: {}".format(OUT_DIR))


if __name__ == "__main__":
    main()
