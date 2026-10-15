import argparse
import random
import warnings

import numpy as np
import torch

from data import GraphDataset
from trainer import ARCZeroTrainer
from utils import read_json_config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="ARCZero clean training/evaluation entrypoint")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--random_shot", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--dims", type=int, default=64)
    parser.add_argument("--json_dir", type=str, default="./best_param_pt")
    parser.add_argument(
        "--train_datasets",
        type=str,
        default="pubmed,Flickr,questions,YelpChi",
        help="Comma-separated train datasets",
    )
    parser.add_argument(
        "--test_datasets",
        type=str,
        default="cora,citeseer,ACM,BlogCatalog,Facebook,weibo,Reddit,Amazon,cs,photo,elliptic,tfinance,tolokers",
        help="Comma-separated test datasets",
    )
    return parser.parse_args()


def default_model_config(random_shot, dims):
    return {
        "in_feats": dims,
        "h_feats": 1024,
        "num_layers": 3,
        "dropout_rate": 0.0,
        "activation": "ELU",
        "num_hops": 2,
        "num_prompt": 10,
        "k": random_shot,
        "t": 2,
        "lr": 5e-5,
        "weight_decay": 1e-5,
    }


def sanitize_model_config(cfg, random_shot, dims):
    defaults = default_model_config(random_shot, dims)
    if cfg is None:
        return defaults

    merged = dict(defaults)
    merged.update(cfg)

    if "drop_rate" in merged and "dropout_rate" not in merged:
        merged["dropout_rate"] = merged["drop_rate"]

    return {
        "in_feats": int(merged["in_feats"]),
        "h_feats": int(merged["h_feats"]),
        "num_layers": int(merged["num_layers"]),
        "dropout_rate": float(merged["dropout_rate"]),
        "activation": str(merged["activation"]),
        "num_hops": int(merged["num_hops"]),
        "num_prompt": int(merged["num_prompt"]),
        "k": int(merged["k"]),
        "t": int(merged["t"]),
        "lr": float(merged["lr"]),
        "weight_decay": float(merged["weight_decay"]),
    }


def aggregate_results(all_results):
    auc_dict = {}
    pr_dict = {}

    for trial_result in all_results:
        for name, metric in trial_result.items():
            auc_dict.setdefault(name, []).append(metric["AUROC"])
            pr_dict.setdefault(name, []).append(metric["AUPRC"])

    output = {}
    for name in auc_dict:
        output[name] = {
            "AUROC_mean": float(np.mean(auc_dict[name])),
            "AUROC_std": float(np.std(auc_dict[name])),
            "AUPRC_mean": float(np.mean(pr_dict[name])),
            "AUPRC_std": float(np.std(pr_dict[name])),
        }
    return output


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()

    train_datasets = [x.strip() for x in args.train_datasets.split(",") if x.strip()]
    test_datasets = [x.strip() for x in args.test_datasets.split(",") if x.strip()]

    raw_model_config = read_json_config("ARC_Kmeans_feat", args.random_shot, args.json_dir)
    model_config = sanitize_model_config(raw_model_config, args.random_shot, args.dims)

    train_data = [GraphDataset(args.dims, name) for name in train_datasets]
    test_data = [GraphDataset(args.dims, name) for name in test_datasets]

    for d in train_data:
        d.propagated(model_config["num_hops"], args.device)
    for d in test_data:
        d.propagated(model_config["num_hops"], args.device)

    train_config = {
        "device": args.device,
        "epochs": args.epochs,
        "testdsets": test_datasets,
    }

    all_results = []
    for seed in range(args.trials):
        set_seed(seed)
        for d in test_data:
            d.kmeans_shot(args.random_shot, seed=0)

        trainer = ARCZeroTrainer(
            train_config=train_config,
            model_config=model_config,
            data={"train": train_data, "test": test_data},
        )
        all_results.append(trainer.train_and_evaluate())

    summary = aggregate_results(all_results)
    for name, metric in summary.items():
        print("-" * 20, name, "-" * 20)
        print(
            "AUROC:{:.4f}+-{:.4f}, AUPRC:{:.4f}+-{:.4f}".format(
                metric["AUROC_mean"],
                metric["AUROC_std"],
                metric["AUPRC_mean"],
                metric["AUPRC_std"],
            )
        )


if __name__ == "__main__":
    main()
