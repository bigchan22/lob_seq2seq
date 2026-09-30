import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main_classical_baselines import make_flat_samples
from main_parallel_asset_attention import bake_tensors, chronological_split_indices
from src.data_loader.preprocess import get_ticker_list


DATA_DIR = Path("./data/processed")
OUT_DIR = Path("./results/classical_baselines_nofuture_valtest_20260713")
SEED = 42


def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0),
    }


def fit_rf(x_train, y_train):
    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=18,
        min_samples_leaf=20,
        n_jobs=8,
        random_state=SEED,
    )
    clf.fit(x_train, y_train)
    return clf


def hash_rows(x, decimals=6):
    rounded = np.round(x, decimals=decimals)
    return {row.tobytes() for row in rounded}


def run_variant(name, x_train, y_train, x_test, y_test, rng, shuffle_labels=False):
    labels = y_train.copy()
    if shuffle_labels:
        labels = labels.copy()
        rng.shuffle(labels)
    clf = fit_rf(x_train, labels)
    train_pred = clf.predict(x_train)
    test_pred = clf.predict(x_test)
    return {
        "variant": name,
        "train": metrics(y_train, train_pred),
        "test": metrics(y_test, test_pred),
    }


def main():
    rng = np.random.default_rng(SEED)
    tickers = get_ticker_list(DATA_DIR)
    full_x, full_y, common_dates = bake_tensors(
        DATA_DIR,
        tickers,
        include_investor=True,
        epsilon=1e-4,
        device=torch.device("cpu"),
    )
    train_end, val_end = chronological_split_indices(len(full_x), 0.7, 0.1)
    features, labels, asset_ids, day_ids = make_flat_samples(full_x, full_y, len(tickers))

    train_mask = day_ids < train_end
    val_mask = (day_ids >= train_end) & (day_ids < val_end)
    test_mask = day_ids >= val_end

    x_train = features[train_mask]
    y_train = labels[train_mask]
    x_val = features[val_mask]
    y_val = labels[val_mask]
    x_test = features[test_mask]
    y_test = labels[test_mask]

    num_raw_features = full_x.shape[-1]
    feature_slices = {
        "full_plus_asset_id": np.arange(features.shape[1]),
        "no_asset_id": np.arange(num_raw_features),
        "price_only": np.arange(0, 3),
        "quantity_only": np.arange(3, 13),
        "investor_only": np.arange(13, num_raw_features),
        "asset_id_only": np.arange(num_raw_features, features.shape[1]),
        "no_price_features": np.arange(3, features.shape[1]),
    }

    train_hashes = hash_rows(x_train[:, :num_raw_features])
    test_hashes = hash_rows(x_test[:, :num_raw_features])

    report = {
        "num_days": len(common_dates),
        "train_days": train_end,
        "validation_days": val_end - train_end,
        "test_days": len(common_dates) - val_end,
        "train_date_range": [str(common_dates[0]), str(common_dates[train_end - 1])],
        "validation_date_range": [str(common_dates[train_end]), str(common_dates[val_end - 1])],
        "test_date_range": [str(common_dates[val_end]), str(common_dates[-1])],
        "sample_counts": {
            "train": int(train_mask.sum()),
            "validation": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "exact_rounded_feature_overlap_train_test": len(train_hashes.intersection(test_hashes)),
        "class_counts": {
            "train": np.bincount(y_train, minlength=3).tolist(),
            "validation": np.bincount(y_val, minlength=3).tolist(),
            "test": np.bincount(y_test, minlength=3).tolist(),
        },
        "variants": [],
    }

    for name, cols in feature_slices.items():
        report["variants"].append(
            run_variant(name, x_train[:, cols], y_train, x_test[:, cols], y_test, rng)
        )

    report["variants"].append(
        run_variant(
            "full_plus_asset_id_shuffled_train_labels",
            x_train,
            y_train,
            x_test,
            y_test,
            rng,
            shuffle_labels=True,
        )
    )

    clf = fit_rf(x_train, y_train)
    report["validation_with_full_plus_asset_id"] = metrics(y_val, clf.predict(x_val))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "random_forest_audit.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
