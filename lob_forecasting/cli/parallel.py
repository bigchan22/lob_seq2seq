import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from lob_forecasting.data.dataset import LOBDataset
from lob_forecasting.data.preprocess import get_ticker_list
from lob_forecasting.data.transforms import LOBFeatureTransformer
from lob_forecasting.models.transformers import AssetAwareLOBTransformer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0),
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def chronological_split_indices(num_days, train_ratio, val_ratio):
    if not 0 < train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1.")
    if not 0 <= val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("--train-ratio + --val-ratio must leave a non-empty test split.")

    train_end = int(num_days * train_ratio)
    val_end = int(num_days * (train_ratio + val_ratio))
    if train_end <= 0 or val_end <= train_end or val_end >= num_days:
        raise ValueError(
            "Invalid chronological split; need non-empty train, validation, and test splits."
        )
    return train_end, val_end


def investor_raw_columns():
    cols = []
    for grp in ["11", "12", "21", "22"]:
        cols.extend([f"BUY_QTY{grp}", f"SELL_QTY{grp}", f"BUY_PRC{grp}", f"SELL_PRC{grp}"])
    return cols


def lag_investor_columns(df, lag_steps):
    if lag_steps <= 0:
        return df
    cols = [col for col in investor_raw_columns() if col in df.columns]
    if cols:
        df.loc[:, cols] = df.loc[:, cols].shift(lag_steps).fillna(0)
    return df


def bake_tensors(data_dir, tickers, include_investor, epsilon, device, investor_lag_steps=0):
    temp_ds = LOBDataset(data_dir, tickers, epsilon=epsilon)
    transformer = LOBFeatureTransformer(temp_ds.col_map, include_investor=include_investor)
    skeleton = pd.DataFrame({"TIME_INTERVAL": temp_ds.master_grid})
    ticker_frames = {}

    for ticker in tqdm(tickers, desc="Reading ticker CSVs", leave=False):
        df = pd.read_csv(data_dir / f"{ticker}.csv")
        df["TIME_INTERVAL"] = df["TIME_INTERVAL"].astype(str).str.strip()
        ticker_frames[ticker] = {date: group for date, group in df.groupby("ORD_DD", sort=False)}

    xs, ys = [], []
    for target_date in tqdm(temp_ds.common_dates, desc=f"Baking {len(tickers)} asset(s)", leave=False):
        day_features, day_labels = [], []
        for ticker in tickers:
            df_day = ticker_frames[ticker][target_date]
            df_aligned = pd.merge(skeleton, df_day, on="TIME_INTERVAL", how="left")
            # Forward-fill only: backward-fill would use later same-day values
            # to impute earlier inputs, which leaks future information.
            df_aligned = df_aligned.ffill().fillna(0)
            if include_investor:
                df_aligned = lag_investor_columns(df_aligned, investor_lag_steps)

            raw = df_aligned[temp_ds.raw_cols].to_numpy(dtype=np.float32)
            ask1 = df_aligned["ASK_STEP1_BSTORD_PRC"].to_numpy(dtype=np.float32)
            bid1 = df_aligned["BID_STEP1_BSTORD_PRC"].to_numpy(dtype=np.float32)
            midprices = (ask1 + bid1) / 2.0

            labels = []
            for t in range(len(midprices) - 1):
                m_t, m_next = midprices[t], midprices[t + 1]
                if m_t > 0:
                    relative_change = (m_next - m_t) / m_t
                    if relative_change > epsilon:
                        labels.append(2)
                    elif relative_change < -epsilon:
                        labels.append(0)
                    else:
                        labels.append(1)
                else:
                    labels.append(1)

            day_features.append(raw[:-1])
            day_labels.append(np.asarray(labels, dtype=np.int64))

        raw_x = torch.from_numpy(np.stack(day_features, axis=0).transpose(1, 0, 2))
        x = torch.nan_to_num(transformer(raw_x))
        y = torch.from_numpy(np.stack(day_labels, axis=0).transpose(1, 0)).long()
        xs.append(x)
        ys.append(y)

    return torch.stack(xs).to(device), torch.stack(ys).long().to(device), temp_ds.common_dates


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features)
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds)
            all_labels.append(labels)

    pred = torch.cat(all_preds, dim=0).detach().cpu().numpy()
    true = torch.cat(all_labels, dim=0).detach().cpu().numpy()
    return true, pred


def parse_args():
    parser = argparse.ArgumentParser(description="Asset-aware parallel LOB Transformer experiment.")
    parser.add_argument("--data-dir", type=Path, default=Path("./data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("./results/parallel_asset_attention"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--include-investor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--investor-lag-steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=16)
    parser.add_argument("--temporal-layers", type=int, default=4)
    parser.add_argument("--cross-asset-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    tickers = get_ticker_list(args.data_dir)
    full_x, full_y, common_dates = bake_tensors(
        args.data_dir,
        tickers,
        args.include_investor,
        args.epsilon,
        device,
        args.investor_lag_steps,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["data_dir"] = str(config["data_dir"])
    config["output_dir"] = str(config["output_dir"])
    config["tickers"] = tickers
    config["num_common_dates"] = len(common_dates)
    config["num_features"] = full_x.shape[-1]
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))

    train_end, val_end = chronological_split_indices(len(full_x), args.train_ratio, args.val_ratio)
    train_loader = DataLoader(
        TensorDataset(full_x[:train_end], full_y[:train_end]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(full_x[train_end:val_end], full_y[train_end:val_end]),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(full_x[val_end:], full_y[val_end:]),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = AssetAwareLOBTransformer(
        num_assets=len(tickers),
        num_features=full_x.shape[-1],
        d_model=args.d_model,
        nhead=args.nhead,
        temporal_layers=args.temporal_layers,
        cross_asset_layers=args.cross_asset_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_f1 = 0.0
    best_epoch = 0
    best_state = None
    history_rows = []

    epoch_iter = range(1, args.epochs + 1)
    if not args.no_progress:
        epoch_iter = tqdm(epoch_iter, desc="Training asset-aware parallel model")

    for epoch in epoch_iter:
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits.reshape(-1, 3), labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        true, pred = evaluate(model, val_loader)
        metrics = classification_metrics(true.reshape(-1), pred.reshape(-1))
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(len(train_loader), 1),
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
        }
        history_rows.append(row)

        if metrics["macro_f1"] > best_f1 or best_state is None:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            ckpt_path = args.output_dir / "best_asset_aware_parallel.pth"
            torch.save(best_state, ckpt_path)

        if args.no_progress and (epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs):
            print(
                f"epoch {epoch:04d}: loss={row['train_loss']:.4f} "
                f"val_acc={metrics['accuracy']:.4f} val_f1={metrics['macro_f1']:.4f} "
                f"best_val_f1={best_f1:.4f} best_epoch={best_epoch}"
            )

        write_csv(
            args.output_dir / "history.csv",
            history_rows,
            ["epoch", "train_loss", "val_accuracy", "val_macro_f1"],
        )
        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    model.load_state_dict(best_state)
    true, pred = evaluate(model, test_loader)
    global_metrics = classification_metrics(true.reshape(-1), pred.reshape(-1))

    asset_rows = []
    for asset_idx, ticker in enumerate(tickers):
        metrics = classification_metrics(true[:, :, asset_idx].reshape(-1), pred[:, :, asset_idx].reshape(-1))
        asset_rows.append(
            {
                "ticker": ticker,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
        )

    summary_rows = [
        {
            "model": "asset_aware_parallel",
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_f1,
            "accuracy": global_metrics["accuracy"],
            "macro_f1": global_metrics["macro_f1"],
            "mean_asset_accuracy": float(np.mean([r["accuracy"] for r in asset_rows])),
            "mean_asset_macro_f1": float(np.mean([r["macro_f1"] for r in asset_rows])),
        }
    ]
    write_csv(args.output_dir / "asset_metrics.csv", asset_rows, ["ticker", "accuracy", "macro_f1"])
    write_csv(
        args.output_dir / "summary.csv",
        summary_rows,
        [
            "model",
            "best_epoch",
            "best_val_macro_f1",
            "accuracy",
            "macro_f1",
            "mean_asset_accuracy",
            "mean_asset_macro_f1",
        ],
    )
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
