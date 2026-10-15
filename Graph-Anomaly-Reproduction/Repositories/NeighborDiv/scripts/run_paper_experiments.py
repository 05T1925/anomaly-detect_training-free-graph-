"""Plan or execute the target-graph main/sampling/ablation experiments.

Default behavior prints the plan only. --execute is required to score graphs.
Author code, unpublished preprocessing and full baseline training are not supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]


def suite_variants(suite, dataset, seeds, default_rank):
    base = {"rank": default_rank, "max_pairs": None, "diversity_statistic": "variance",
            "reference": "median", "low_degree_fallback": "zero", "evaluation_scope": "all"}
    variants = []
    if suite == "main":
        variants = [("full", {}, [0])]
    elif suite == "sampling":
        variants = [("full", {}, [0]), ("sample-100", {"max_pairs": 100}, seeds),
                    ("sample-50", {"max_pairs": 50}, seeds)]
    elif suite == "rank":
        if dataset["name"] not in ("Cora", "YelpChi", "Reddit", "Questions"):
            return
        variants = [("rank-{}".format(r), {"rank": r}, [0]) for r in (8, 16, 32, 64)
                    if r <= dataset["paper_features"]]
    elif suite == "low-degree":
        variants = [("zero", {}, [0]), ("median", {"low_degree_fallback": "median"}, [0]),
                    ("valid-only", {"evaluation_scope": "valid"}, [0])]
    elif suite == "statistic":
        variants = [("variance-median", {}, [0]), ("variance-mean", {"reference": "mean"}, [0]),
                    ("std-median", {"diversity_statistic": "std"}, [0]),
                    ("mean-median", {"diversity_statistic": "mean"}, [0])]
    for name, overrides, repetitions in variants:
        for seed in repetitions:
            yield name, seed, dict(base, **overrides)


def build_plan(args, config):
    defaults = config["defaults"]
    datasets = config["datasets"]
    requested = args.datasets if args.datasets is not None else [d["name"] for d in datasets]
    unknown = set(requested) - {d["name"] for d in datasets}
    if unknown:
        raise ValueError("Unknown datasets: {}".format(", ".join(sorted(unknown))))
    seeds = args.seeds if args.seeds is not None else config["sampling_seeds"]
    if not seeds or len(seeds) != len(set(seeds)) or any(s < 0 or s >= 2**32 for s in seeds):
        raise ValueError("Sampling seeds must be distinct integers in [0, 2**32).")
    plan = []
    for dataset in datasets:
        if dataset["name"] not in requested:
            continue
        for variant, pair_seed, settings in suite_variants(args.suite, dataset, seeds, defaults["rank"]):
            data_path = Path(args.data_root).expanduser().resolve() / dataset["path"]
            out_dir = Path(args.output_root).expanduser().resolve() / args.suite / dataset["name"] / variant / "seed-{}".format(pair_seed)
            cli = ["--data", str(data_path), "--format", dataset["format"],
                   "--edge-policy", dataset["edge_policy"], "--output-dir", str(out_dir),
                   "--full-backend", args.full_backend,
                   "--rank", str(settings["rank"]), "--svd-random-state", str(args.svd_seed),
                   "--pair-random-state", str(pair_seed), "--svd-algorithm", args.svd_algorithm,
                   "--svd-n-iter", str(defaults["svd_n_iter"]),
                   "--pair-batch-size", str(defaults["pair_batch_size"]),
                   "--threshold-lambda", str(defaults["threshold_lambda"]),
                   "--diversity-statistic", settings["diversity_statistic"],
                   "--reference", settings["reference"], "--low-degree-fallback", settings["low_degree_fallback"],
                   "--evaluation-scope", settings["evaluation_scope"]]
            if settings["max_pairs"] is not None:
                cli += ["--max-pairs", str(settings["max_pairs"])]
            plan.append({"dataset": dataset["name"], "variant": variant, "pair_seed": pair_seed,
                         "reference": dataset, "data_exists": data_path.is_file(),
                         "output_dir": str(out_dir), "arguments": cli})
    if not plan:
        raise ValueError("Selected suite/datasets produced no runnable configurations.")
    return plan


def write_summary(folder, rows):
    (folder / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    fields = ("dataset", "variant", "pair_seed", "status", "AUROC", "AP", "paper_auc", "paper_ap",
              "delta_auc", "delta_ap", "n_nodes", "valid_nodes", "paper_valid_nodes",
              "scoring_seconds", "load_and_scoring_seconds", "output_dir", "data_note", "error")
    with (folder / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(plan, rows):
    groups = {}
    for item in plan:
        groups.setdefault((item["dataset"], item["variant"]), {"expected_runs": 0, "rows": []})["expected_runs"] += 1
    for row in rows:
        groups[(row["dataset"], row["variant"])]["rows"].append(row)
    result = []
    for (dataset, variant), group in groups.items():
        completed = [r for r in group["rows"] if r["status"] == "completed"]
        entry = {"dataset": dataset, "variant": variant, "expected_runs": group["expected_runs"],
                 "completed_runs": len(completed), "complete": len(completed) == group["expected_runs"]}
        for metric in ("AUROC", "AP", "scoring_seconds", "load_and_scoring_seconds"):
            values = [r[metric] for r in completed if r[metric] is not None]
            entry[metric + "_mean"] = statistics.mean(values) if values else None
            entry[metric + "_std_population"] = statistics.pstdev(values) if values else None
        result.append(entry)
    # Never describe an incomplete/subset average as the seven-target result.
    seven_means = {}
    target_names = {"Cora", "YelpChi", "Reddit", "T-Finance", "Tolokers", "Disney", "Questions"}
    for variant in {item["variant"] for item in result}:
        subset = [item for item in result if item["variant"] == variant and item["complete"]]
        if {item["dataset"] for item in subset} == target_names:
            seven_means[variant] = {m: statistics.mean([item[m + "_mean"] for item in subset])
                                    for m in ("AUROC", "AP") if all(item[m + "_mean"] is not None for item in subset)}
    return {"per_dataset": result, "complete_seven_target_averages": seven_means,
            "seed_std_note": "population std over specified pair seeds with fixed SVD seed; not source-domain std in Table 2"}


def main():
    parser = argparse.ArgumentParser(description="Plan NeighborDiv experiments; add --execute only when ready to run.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/paper_experiments.json"))
    parser.add_argument("--suite", choices=("main", "sampling", "rank", "low-degree", "statistic"), default="main")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--data-root", default=str(WORKSPACE_ROOT / "data"))
    parser.add_argument("--output-root", default=str(WORKSPACE_ROOT / "results/NeighborDiv/paper-audit"))
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--svd-seed", type=int, default=None)
    parser.add_argument("--svd-algorithm", choices=("randomized", "arpack"), default=None)
    parser.add_argument("--full-backend", choices=("stream", "moments"), default="stream")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--keep-going", action="store_true", help="Record an error and continue other runs.")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    args.svd_seed = config["defaults"]["svd_random_state"] if args.svd_seed is None else args.svd_seed
    args.svd_algorithm = config["defaults"]["svd_algorithm"] if args.svd_algorithm is None else args.svd_algorithm
    if not 0 <= args.svd_seed < 2**32:
        parser.error("--svd-seed must be in [0, 2**32).")
    try:
        plan = build_plan(args, config)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.execute:
        print(json.dumps({"mode": "plan only; no scoring or output writes", "suite": args.suite, "runs": plan},
                         indent=2, ensure_ascii=False))
        return

    folder = Path(args.output_root).expanduser().resolve() / args.suite
    if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
        parser.error("Suite output directory is non-empty; choose a new --output-root.")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Delayed import: planning itself needs only the standard library.
    from run_neighbordiv import build_parser, run_experiment
    runner_parser = build_parser()
    rows = []
    for item in plan:
        ref = item["reference"]
        row = {"dataset": item["dataset"], "variant": item["variant"], "pair_seed": item["pair_seed"],
               "status": "pending", "AUROC": None, "AP": None, "paper_auc": ref["paper_auc"], "paper_ap": ref["paper_ap"],
               "delta_auc": None, "delta_ap": None, "n_nodes": None, "valid_nodes": None,
               "paper_valid_nodes": ref["paper_valid_nodes"], "scoring_seconds": None, "load_and_scoring_seconds": None,
               "output_dir": item["output_dir"], "data_note": ref["data_note"], "error": ""}
        try:
            run = run_experiment(runner_parser.parse_args(item["arguments"]))
            row.update({"status": "completed", "AUROC": run["metrics"]["AUROC"], "AP": run["metrics"]["AP"],
                        "n_nodes": run["graph"]["n_nodes"], "valid_nodes": run["graph"]["n_valid_nodes_degree_ge_2"],
                        "scoring_seconds": run["scoring_seconds"], "load_and_scoring_seconds": run["load_and_scoring_seconds"]})
            # Main references are displayed only for the main/default Full setting.
            if item["variant"] in ("full", "rank-8", "zero", "variance-median"):
                if row["AUROC"] is not None:
                    row["delta_auc"] = row["AUROC"] - ref["paper_auc"]
                if row["AP"] is not None:
                    row["delta_ap"] = row["AP"] - ref["paper_ap"]
        except Exception as exc:
            row.update({"status": "failed", "error": "{}: {}".format(type(exc).__name__, exc)})
            rows.append(row)
            write_summary(folder, rows)
            (folder / "aggregate.json").write_text(json.dumps(aggregate(plan, rows), indent=2) + "\n", encoding="utf-8")
            if not args.keep_going:
                raise
            print(row["error"], file=sys.stderr)
            continue
        rows.append(row)
        write_summary(folder, rows)
    (folder / "aggregate.json").write_text(json.dumps(aggregate(plan, rows), indent=2) + "\n", encoding="utf-8")
    if any(row["status"] == "failed" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
