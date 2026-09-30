import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lob_forecasting.cli.parallel import bake_tensors, chronological_split_indices
from lob_forecasting.data.preprocess import get_ticker_list


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0),
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_flat_samples(full_x, full_y, num_assets):
    x_np = full_x.detach().cpu().numpy()
    y_np = full_y.detach().cpu().numpy()
    num_days, seq_len, _, num_features = x_np.shape

    features = x_np.reshape(num_days * seq_len * num_assets, num_features)
    labels = y_np.reshape(num_days * seq_len * num_assets)
    asset_ids = np.tile(np.arange(num_assets), num_days * seq_len)
    eye = np.eye(num_assets, dtype=np.float32)
    features_with_asset = np.concatenate([features, eye[asset_ids]], axis=1)
    day_ids = np.repeat(np.arange(num_days), seq_len * num_assets)
    return features_with_asset, labels, asset_ids, day_ids


def predict_previous_label(train_y, test_y):
    train_flat = train_y.detach().cpu().numpy().reshape(-1)
    fallback = int(np.bincount(train_flat, minlength=3).argmax())
    y_np = test_y.detach().cpu().numpy()
    pred = np.empty_like(y_np)
    pred[:, 0, :] = fallback
    pred[:, 1:, :] = y_np[:, :-1, :]
    return y_np.reshape(-1), pred.reshape(-1)


def build_classifier(name, args):
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "stratified":
        return DummyClassifier(strategy="stratified", random_state=args.seed)
    if name == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=args.max_iter,
                C=args.logistic_c,
                class_weight=args.class_weight,
                n_jobs=args.n_jobs,
                random_state=args.seed,
            ),
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=args.rf_trees,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            class_weight=args.class_weight,
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=tuple(args.mlp_hidden),
                activation="relu",
                alpha=args.mlp_alpha,
                batch_size=args.mlp_batch_size,
                learning_rate_init=args.mlp_learning_rate,
                max_iter=args.max_iter,
                early_stopping=True,
                random_state=args.seed,
            ),
        )
    raise ValueError(f"Unknown baseline model: {name}")


def evaluate_flat_predictions(model_name, tickers, y_true, y_pred, asset_ids):
    global_metrics = classification_metrics(y_true, y_pred)
    summary = {
        "model": model_name,
        "accuracy": global_metrics["accuracy"],
        "balanced_accuracy": global_metrics["balanced_accuracy"],
        "macro_f1": global_metrics["macro_f1"],
    }

    asset_rows = []
    for asset_idx, ticker in enumerate(tickers):
        mask = asset_ids == asset_idx
        metrics = classification_metrics(y_true[mask], y_pred[mask])
        asset_rows.append(
            {
                "model": model_name,
                "ticker": ticker,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
        )

    summary["mean_asset_accuracy"] = float(np.mean([r["accuracy"] for r in asset_rows]))
    summary["mean_asset_balanced_accuracy"] = float(np.mean([r["balanced_accuracy"] for r in asset_rows]))
    summary["mean_asset_macro_f1"] = float(np.mean([r["macro_f1"] for r in asset_rows]))
    return summary, asset_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Classical baselines for multi-asset LOB prediction.")
    parser.add_argument("--data-dir", type=Path, default=Path("./data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("./results/classical_baselines"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["majority", "persistence", "logistic", "random_forest", "mlp"],
        choices=["majority", "stratified", "persistence", "logistic", "random_forest", "mlp"],
    )
    parser.add_argument("--include-investor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--investor-lag-steps", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--class-weight", choices=["balanced"], default=None)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--rf-trees", type=int, default=300)
    parser.add_argument("--rf-max-depth", type=int, default=18)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=20)
    parser.add_argument("--mlp-hidden", nargs="+", type=int, default=[128, 64])
    parser.add_argument("--mlp-alpha", type=float, default=1e-4)
    parser.add_argument("--mlp-batch-size", type=int, default=2048)
    parser.add_argument("--mlp-learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tickers = get_ticker_list(args.data_dir)
    if args.max_tickers is not None:
        tickers = tickers[: args.max_tickers]

    full_x, full_y, common_dates = bake_tensors(
        args.data_dir,
        tickers,
        args.include_investor,
        args.epsilon,
        torch.device("cpu"),
        args.investor_lag_steps,
    )
    train_end, val_end = chronological_split_indices(len(full_x), args.train_ratio, args.val_ratio)

    features, labels, asset_ids, day_ids = make_flat_samples(full_x, full_y, len(tickers))
    train_mask = day_ids < train_end
    test_mask = day_ids >= val_end

    config = vars(args).copy()
    config["data_dir"] = str(config["data_dir"])
    config["output_dir"] = str(config["output_dir"])
    config["tickers"] = tickers
    config["num_common_dates"] = len(common_dates)
    config["num_features_without_asset_id"] = int(full_x.shape[-1])
    config["num_features_with_asset_id"] = int(features.shape[-1])
    config["train_days"] = int(train_end)
    config["validation_days"] = int(val_end - train_end)
    config["test_days"] = int(len(full_x) - val_end)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))

    summary_rows = []
    asset_rows = []

    for model_name in args.models:
        print(f"Running {model_name} baseline...")
        if model_name == "persistence":
            y_true, y_pred = predict_previous_label(full_y[:train_end], full_y[val_end:])
            repeated_asset_ids = asset_ids[test_mask]
        else:
            clf = build_classifier(model_name, args)
            clf.fit(features[train_mask], labels[train_mask])
            y_true = labels[test_mask]
            y_pred = clf.predict(features[test_mask])
            repeated_asset_ids = asset_ids[test_mask]

        summary, per_asset = evaluate_flat_predictions(
            model_name,
            tickers,
            y_true,
            y_pred,
            repeated_asset_ids,
        )
        summary_rows.append(summary)
        asset_rows.extend(per_asset)

        write_csv(
            args.output_dir / "summary.csv",
            summary_rows,
            [
                "model",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "mean_asset_accuracy",
                "mean_asset_balanced_accuracy",
                "mean_asset_macro_f1",
            ],
        )
        write_csv(
            args.output_dir / "asset_metrics.csv",
            asset_rows,
            ["model", "ticker", "accuracy", "balanced_accuracy", "macro_f1"],
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
