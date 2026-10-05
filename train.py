import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hobrp.training import _to_jsonable_result, train_one_fold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_root", required=True)
    parser.add_argument("--node_feature_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--negative_node_feature_path")
    parser.add_argument("--folds", required=True)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--negative_topk", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--selection_metric", choices=["auc", "aupr"], default="aupr")
    parser.add_argument("--propagation_steps", type=int, default=10)
    parser.add_argument("--propagation_decay", type=float, default=0.1)
    args = parser.parse_args()

    split_root = Path(args.split_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_ids = [int(value.strip()) for value in args.folds.split(",") if value.strip()]
    results = []

    for fold_id in fold_ids:
        fold_dir = split_root / f"fold_{fold_id}"
        model, result = train_one_fold(
            fold_dir=fold_dir,
            node_feature_path=args.node_feature_path,
            neg_node_feature_path=args.negative_node_feature_path,
            order=args.order,
            negative_topk=args.negative_topk,
            device=args.device,
            seed=args.seed + fold_id,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            selection_metric=args.selection_metric,
            K=args.propagation_steps,
            alpha=args.propagation_decay,
        )
        fold_output = output_dir / f"fold_{fold_id}"
        fold_output.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), fold_output / "best_model_hobrp.pt")
        with (fold_output / "result_hobrp.json").open("w", encoding="utf-8") as file:
            json.dump(_to_jsonable_result(result), file, indent=2)
        metric_row = {key: value for key, value in result.items() if np.isscalar(value) and key not in {"fold_dir", "node_feature_path", "neg_node_feature_path"}}
        metric_row["fold"] = fold_id
        results.append(metric_row)

    frame = pd.DataFrame(results)
    frame.to_csv(output_dir / "metrics_all_folds.csv", index=False)
    numeric = frame.select_dtypes(include=[np.number]).drop(columns=["fold"], errors="ignore")
    summary = pd.DataFrame({"mean": numeric.mean(), "std": numeric.std(ddof=1)}).reset_index(names="metric")
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)


if __name__ == "__main__":
    main()
