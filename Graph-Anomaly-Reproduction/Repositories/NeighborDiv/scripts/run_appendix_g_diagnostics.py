"""Reproduce post-hoc Appendix G diagnostics from saved T-Finance scores.

Labels are loaded only for this analysis after each anomaly score file exists;
they are never passed to the scoring implementation.
"""

from __future__ import annotations

import csv
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_graph  # noqa: E402


FULL_DIR = REPO_ROOT / "results/NeighborDiv/paper-audit-tfinance-full-retry/main/T-Finance/full/seed-0"
SAMPLE_ROOT = REPO_ROOT / "results/NeighborDiv/paper-audit/sampling/T-Finance"
OUTPUT_DIR = REPO_ROOT / "results/NeighborDiv/paper-audit-appendix-g/T-Finance"
DATA_PATH = REPO_ROOT / "data/T-Finance/tfinance-source.zip"
TOP_K = (100, 500, 1000, 5000)


def read_scores(folder: Path, labels: np.ndarray) -> np.ndarray:
    values = np.loadtxt(folder / "anomaly_scores.csv", delimiter=",", skiprows=1)
    if values.ndim != 2 or values.shape != (labels.size, 2):
        raise ValueError("Unexpected anomaly_scores.csv shape in {}".format(folder))
    if not np.array_equal(values[:, 0].astype(np.int64), np.arange(labels.size)):
        raise ValueError("Node IDs are not in original order in {}".format(folder))
    scores = values[:, 1].astype(np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite anomaly score in {}".format(folder))
    return scores


def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    top = np.argpartition(scores, scores.size - k)[-k:]
    return float(labels[top].mean())


def pr_at_grid(labels: np.ndarray, scores: np.ndarray, recall_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    precision, recall, _ = precision_recall_curve(labels, scores)
    # Collapse repeated recall values before interpolation; keep the best
    # precision attainable at each recall, as is standard for a PR envelope.
    order = np.argsort(recall, kind="mergesort")
    recall, precision = recall[order], precision[order]
    unique_recall = np.unique(recall)
    envelope = np.array([precision[recall == value].max() for value in unique_recall])
    envelope = np.maximum.accumulate(envelope[::-1])[::-1]
    return unique_recall, np.interp(recall_grid, unique_recall, envelope)


def _line_path(xs, ys, x0, y0, width, height, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0):
    points = []
    for x, y in zip(xs, ys):
        px = x0 + (float(x) - xmin) / (xmax - xmin) * width
        py = y0 + height - (float(y) - ymin) / (ymax - ymin) * height
        points.append("{:.2f},{:.2f}".format(px, py))
    return "M " + " L ".join(points)


def write_pr_svg(path: Path, summary: dict, prevalence: float):
    width, height, x0, y0, plot_w, plot_h = 740, 520, 70, 35, 610, 400
    grid = np.asarray(summary["precision_recall_curve"]["recall_grid"], dtype=float)
    curves = summary["precision_recall_curve"]["curves"]
    colors = {"Full": "#1f77b4", "k=100": "#ff7f0e", "k=50": "#2ca02c"}
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(width, height, width, height),
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="370" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">Precision–Recall on T-Finance</text>',
    ]
    for tick in np.linspace(0.0, 1.0, 6):
        x = x0 + tick * plot_w
        y = y0 + plot_h - tick * plot_h
        parts.append('<line x1="{:.2f}" y1="{}" x2="{:.2f}" y2="{}" stroke="#dddddd"/>'.format(x, y0, x, y0 + plot_h))
        parts.append('<line x1="{}" y1="{:.2f}" x2="{}" y2="{:.2f}" stroke="#dddddd"/>'.format(x0, y, x0 + plot_w, y))
        parts.append('<text x="{:.2f}" y="{:.2f}" text-anchor="middle" font-family="sans-serif" font-size="12">{:.1f}</text>'.format(x, y0 + plot_h + 20, tick))
        parts.append('<text x="{}" y="{:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{:.1f}</text>'.format(x0 - 8, y + 4, tick))
    y_rand = y0 + plot_h - prevalence * plot_h
    parts.append('<line x1="{}" y1="{:.2f}" x2="{}" y2="{:.2f}" stroke="#777777" stroke-dasharray="6 5"/>'.format(x0, y_rand, x0 + plot_w, y_rand))
    for name, color in colors.items():
        curve = curves[name]
        mean = np.asarray(curve["mean_precision"], dtype=float)
        std = np.asarray(curve["std_precision_population"], dtype=float)
        if name != "Full":
            upper = np.minimum(1.0, mean + std)
            lower = np.maximum(0.0, mean - std)
            upper_path = _line_path(grid, upper, x0, y0, plot_w, plot_h)
            lower_path = _line_path(grid[::-1], lower[::-1], x0, y0, plot_w, plot_h)
            parts.append('<path d="{} {} Z" fill="{}" fill-opacity="0.14" stroke="none"/>'.format(upper_path, "L" + lower_path[1:], color))
        parts.append('<path d="{}" fill="none" stroke="{}" stroke-width="2.2"/>'.format(
            _line_path(grid, mean, x0, y0, plot_w, plot_h), color))
    parts.extend([
        '<rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="#333333"/>'.format(x0, y0, plot_w, plot_h),
        '<text x="{}" y="{}" text-anchor="middle" font-family="sans-serif" font-size="14">Recall</text>'.format(x0 + plot_w / 2, y0 + plot_h + 48),
        '<text x="18" y="{}" transform="rotate(-90 18 {})" text-anchor="middle" font-family="sans-serif" font-size="14">Precision</text>'.format(y0 + plot_h / 2, y0 + plot_h / 2),
    ])
    legend = [("Full", colors["Full"]), ("k=100 mean ± SD", colors["k=100"]), ("k=50 mean ± SD", colors["k=50"]), ("Random", "#777777")]
    for index, (label, color) in enumerate(legend):
        x, y = x0 + 18 + (index % 2) * 230, y0 + 20 + (index // 2) * 22
        dash = ' stroke-dasharray="5 4"' if label == "Random" else ""
        parts.append('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2"{} />'.format(x, y, x + 24, y, color, dash))
        parts.append('<text x="{}" y="{}" font-family="sans-serif" font-size="12">{}</text>'.format(x + 31, y + 4, label))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_distribution_svg(path: Path, normal: np.ndarray, anomaly: np.ndarray):
    combined = np.concatenate((normal, anomaly))
    x_min, x_max = np.quantile(combined, [0.005, 0.995])
    hist_n, edges = np.histogram(normal, bins=60, range=(x_min, x_max), density=True)
    hist_a, _ = np.histogram(anomaly, bins=60, range=(x_min, x_max), density=True)
    width, height, x0, y0, plot_w, plot_h = 740, 520, 80, 35, 600, 400
    ymax = float(max(hist_n.max(), hist_a.max())) * 1.08
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(width, height, width, height),
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="370" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">Full NeighborDiv scores on T-Finance</text>',
    ]
    for tick in range(6):
        frac = tick / 5.0
        y = y0 + plot_h - frac * plot_h
        x_val = x_min + frac * (x_max - x_min)
        parts.append('<line x1="{}" y1="{:.2f}" x2="{}" y2="{:.2f}" stroke="#dddddd"/>'.format(x0, y, x0 + plot_w, y))
        parts.append('<text x="{:.2f}" y="{}" text-anchor="middle" font-family="sans-serif" font-size="12">{:.2f}</text>'.format(x0 + frac * plot_w, y0 + plot_h + 20, x_val))
    for values, color in ((hist_n, "#1f77b4"), (hist_a, "#d62728")):
        coords = []
        for idx, value in enumerate(values):
            x = x0 + ((edges[idx] + edges[idx + 1]) * 0.5 - x_min) / (x_max - x_min) * plot_w
            y = y0 + plot_h - value / ymax * plot_h
            coords.append("{:.2f},{:.2f}".format(x, y))
        parts.append('<polyline points="{}" fill="none" stroke="{}" stroke-width="2.2"/>'.format(" ".join(coords), color))
    parts.extend([
        '<rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="#333333"/>'.format(x0, y0, plot_w, plot_h),
        '<text x="{}" y="{}" text-anchor="middle" font-family="sans-serif" font-size="14">Anomaly score (0.5–99.5 percentile range)</text>'.format(x0 + plot_w / 2, y0 + plot_h + 48),
        '<text x="18" y="{}" transform="rotate(-90 18 {})" text-anchor="middle" font-family="sans-serif" font-size="14">Density</text>'.format(y0 + plot_h / 2, y0 + plot_h / 2),
        '<line x1="520" y1="56" x2="544" y2="56" stroke="#1f77b4" stroke-width="2"/><text x="550" y="60" font-family="sans-serif" font-size="12">Normal</text>',
        '<line x1="610" y1="56" x2="634" y2="56" stroke="#d62728" stroke-width="2"/><text x="640" y="60" font-family="sans-serif" font-size="12">Anomaly</text>',
        "</svg>",
    ])
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    OUTPUT_DIR = parser.parse_args().output_dir.resolve()
    graph = load_graph(DATA_PATH, data_format="dgl-zip", edge_policy="preserve", load_labels=True)
    labels = graph.labels.astype(np.int64, copy=False)
    if np.unique(labels).size != 2:
        raise ValueError("Appendix G requires both classes in the evaluation labels.")

    configurations = [("Full", 0, FULL_DIR)]
    for k in (100, 50):
        for seed in range(5):
            configurations.append(("k={}".format(k), seed, SAMPLE_ROOT / "sample-{}".format(k) / "seed-{}".format(seed)))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    score_by_name = {}
    for name, seed, folder in configurations:
        scores = read_scores(folder, labels)
        score_by_name[(name, seed)] = scores
        row = {
            "configuration": name,
            "pair_seed": seed,
            "AUROC": float(roc_auc_score(labels, scores)),
            "AP": float(average_precision_score(labels, scores)),
            "n_nodes": int(labels.size),
            "n_anomalies": int(labels.sum()),
        }
        row.update({"P@{}".format(k): precision_at_k(labels, scores, k) for k in TOP_K})
        metric_rows.append(row)

    metric_fields = ["configuration", "pair_seed", "AUROC", "AP", "n_nodes", "n_anomalies"] + ["P@{}".format(k) for k in TOP_K]
    with (OUTPUT_DIR / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(metric_rows)

    full_scores = score_by_name[("Full", 0)]
    normal_scores = full_scores[labels == 0]
    anomalous_scores = full_scores[labels == 1]
    ks = ks_2samp(normal_scores, anomalous_scores, alternative="two-sided", method="auto")
    curve_grid = np.linspace(0.0, 1.0, 501)
    curve_rows = []
    curve_summary = {}
    for name in ("Full", "k=100", "k=50"):
        seeds = [0] if name == "Full" else list(range(5))
        curves = []
        for seed in seeds:
            scores = score_by_name[(name, seed)]
            precision_values, recall_values, _ = precision_recall_curve(labels, scores)
            for recall_value, precision_value in zip(recall_values, precision_values):
                curve_rows.append({"configuration": name, "pair_seed": seed,
                                   "recall": float(recall_value), "precision": float(precision_value)})
            _, curve = pr_at_grid(labels, scores, curve_grid)
            curves.append(curve)
        matrix = np.vstack(curves)
        curve_summary[name] = {"mean_precision": matrix.mean(axis=0).tolist(),
                               "std_precision_population": matrix.std(axis=0, ddof=0).tolist(),
                               "seeds": seeds}

    with (OUTPUT_DIR / "precision_recall_points.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["configuration", "pair_seed", "recall", "precision"])
        writer.writeheader()
        writer.writerows(curve_rows)

    p_at_k_summary = {}
    for name in ("Full", "k=100", "k=50"):
        rows = [r for r in metric_rows if r["configuration"] == name]
        p_at_k_summary[name] = {
            "pair_seeds": [r["pair_seed"] for r in rows],
            "P_at_K_mean": {"P@{}".format(k): float(np.mean([r["P@{}".format(k)] for r in rows])) for k in TOP_K},
            "P_at_K_std_population": {"P@{}".format(k): float(np.std([r["P@{}".format(k)] for r in rows], ddof=0)) for k in TOP_K},
        }

    summary = {
        "paper_section": "Appendix G, T-Finance post-hoc diagnostics",
        "data_path": str(DATA_PATH),
        "label_use": "labels are used only after loading saved scores, for diagnostic stratification and evaluation",
        "n_nodes": int(labels.size),
        "n_anomalies": int(labels.sum()),
        "anomaly_prevalence": float(labels.mean()),
        "full_score_distribution": {
            "normal_count": int(normal_scores.size),
            "normal_mean": float(normal_scores.mean()),
            "normal_std_population": float(normal_scores.std(ddof=0)),
            "anomaly_count": int(anomalous_scores.size),
            "anomaly_mean": float(anomalous_scores.mean()),
            "anomaly_std_population": float(anomalous_scores.std(ddof=0)),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
        },
        "precision_at_k": p_at_k_summary,
        "precision_recall_curve": {
            "recall_grid": curve_grid.tolist(),
            "curves": curve_summary,
            "csv": "precision_recall_points.csv",
        },
        "saved_score_sources": {name + "/seed-{}".format(seed): str(folder) for name, seed, folder in configurations},
        "notes": [
            "The five sampling seeds are summarized locally; Figure 3 in the paper shows a single curve per configuration.",
            "The loaded release has 1,804 positives; the paper's Appendix G text reports 1,803, so the data versions are not identical.",
            "The paper's baseline-variance values are published reference numbers and were not re-estimated here.",
        ],
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        write_pr_svg(OUTPUT_DIR / "tfinance_pr_curve.svg", summary, float(labels.mean()))
        write_distribution_svg(OUTPUT_DIR / "tfinance_score_distribution.svg", normal_scores, anomalous_scores)
        print("Matplotlib unavailable; wrote vector SVG figures and numerical diagnostics.")
        print("outputs: {}".format(OUTPUT_DIR))
        return

    colors = {"Full": "#1f77b4", "k=100": "#ff7f0e", "k=50": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    for name, curve in curve_summary.items():
        mean_precision = np.asarray(curve["mean_precision"])
        std_precision = np.asarray(curve["std_precision_population"])
        ax.plot(curve_grid, mean_precision, label=name, color=colors[name], linewidth=2)
        if name != "Full":
            ax.fill_between(curve_grid, np.maximum(0, mean_precision - std_precision),
                            np.minimum(1, mean_precision + std_precision), color=colors[name], alpha=0.15)
    ax.axhline(labels.mean(), color="gray", linestyle="--", linewidth=1, label="Random baseline")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1),
           title="Precision–Recall on T-Finance")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.savefig(OUTPUT_DIR / "tfinance_pr_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    ax.hist(normal_scores, bins=60, density=True, alpha=0.58, color="#1f77b4", label="Normal")
    ax.hist(anomalous_scores, bins=60, density=True, alpha=0.58, color="#d62728", label="Anomaly")
    ax.set(xlabel="Anomaly score", ylabel="Density", title="Full NeighborDiv score distribution on T-Finance")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.savefig(OUTPUT_DIR / "tfinance_score_distribution.png", dpi=180)
    plt.close(fig)

    print("diagnostics: {} configurations".format(len(metric_rows)))
    print("Full KS={:.6f}, p={:.6g}".format(ks.statistic, ks.pvalue))
    print("outputs: {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()
