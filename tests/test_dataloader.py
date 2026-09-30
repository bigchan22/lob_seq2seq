import tempfile
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader.dataset import LOBDataset
from src.data_loader.transforms import LOBFeatureTransformer


MASTER_GRID = [
    "09:00:00", "09:10:00", "09:20:00", "09:30:00", "09:40:00", "09:50:00",
    "10:00:00", "10:10:00", "10:20:00", "10:30:00", "10:40:00", "10:50:00",
    "11:00:00", "11:10:00", "11:20:00", "11:30:00", "11:40:00", "11:50:00",
    "12:00:00", "12:10:00", "12:20:00", "12:30:00", "12:40:00", "12:50:00",
    "13:00:00", "13:10:00", "13:20:00", "13:30:00", "13:40:00", "13:50:00",
    "14:00:00", "14:10:00", "14:20:00", "14:30:00", "14:40:00", "14:50:00",
    "15:00:00", "15:10:00", "15:20:00",
]

RAW_COLS = [
    "SELL_PRC11", "BUY_PRC11", "SELL_PRC12", "BUY_PRC12",
    "SELL_PRC21", "BUY_PRC21", "SELL_PRC22", "BUY_PRC22",
    "SELL_QTY11", "BUY_QTY11", "SELL_QTY12", "BUY_QTY12",
    "SELL_QTY21", "BUY_QTY21", "SELL_QTY22", "BUY_QTY22",
    "TRD_PRC",
    "ASK_STEP1_BSTORD_PRC", "ASK_STEP1_BSTORD_RQTY",
    "ASK_STEP2_BSTORD_PRC", "ASK_STEP2_BSTORD_RQTY",
    "ASK_STEP3_BSTORD_PRC", "ASK_STEP3_BSTORD_RQTY",
    "ASK_STEP4_BSTORD_PRC", "ASK_STEP4_BSTORD_RQTY",
    "ASK_STEP5_BSTORD_PRC", "ASK_STEP5_BSTORD_RQTY",
    "BID_STEP1_BSTORD_PRC", "BID_STEP1_BSTORD_RQTY",
    "BID_STEP2_BSTORD_PRC", "BID_STEP2_BSTORD_RQTY",
    "BID_STEP3_BSTORD_PRC", "BID_STEP3_BSTORD_RQTY",
    "BID_STEP4_BSTORD_PRC", "BID_STEP4_BSTORD_RQTY",
    "BID_STEP5_BSTORD_PRC", "BID_STEP5_BSTORD_RQTY",
    "SELL_PRC2_AGG_WAVG", "BUY_PRC2_AGG_WAVG",
    "SELL_QTY2_AGG_SUM", "BUY_QTY2_AGG_SUM",
]


def _write_synthetic_ticker(path: Path, ticker: str, offset: float) -> None:
    rows = []
    for date in ["20240102", "20240103"]:
        for step, interval in enumerate(MASTER_GRID):
            mid = 100.0 + offset + step * 0.01
            row = {
                "ORD_DD": date,
                "TIME_INTERVAL": interval,
                "TRD_PRC": mid,
                "ASK_STEP1_BSTORD_PRC": mid + 0.01,
                "BID_STEP1_BSTORD_PRC": mid - 0.01,
                "SELL_PRC2_AGG_WAVG": mid + 0.03,
                "BUY_PRC2_AGG_WAVG": mid - 0.03,
                "SELL_QTY2_AGG_SUM": 1000 + step,
                "BUY_QTY2_AGG_SUM": 1100 + step,
            }
            for group in ["11", "12", "21", "22"]:
                row[f"SELL_PRC{group}"] = mid + 0.02
                row[f"BUY_PRC{group}"] = mid - 0.02
                row[f"SELL_QTY{group}"] = 100 + step
                row[f"BUY_QTY{group}"] = 120 + step
            for level in range(1, 6):
                row[f"ASK_STEP{level}_BSTORD_PRC"] = mid + level * 0.01
                row[f"ASK_STEP{level}_BSTORD_RQTY"] = 1000 + level + step
                row[f"BID_STEP{level}_BSTORD_PRC"] = mid - level * 0.01
                row[f"BID_STEP{level}_BSTORD_RQTY"] = 900 + level + step
            rows.append(row)

    pd.DataFrame(rows, columns=["ORD_DD", "TIME_INTERVAL", *RAW_COLS]).to_csv(
        path / f"{ticker}.csv",
        index=False,
    )


def test_pipeline():
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp)
        tickers = ["AAA", "BBB"]
        _write_synthetic_ticker(data_path, tickers[0], 0.0)
        _write_synthetic_ticker(data_path, tickers[1], 1.0)

        raw_dataset = LOBDataset(data_path, tickers)
        transform = LOBFeatureTransformer(raw_dataset.col_map, include_investor=True)
        dataset = LOBDataset(data_path, tickers, transform=transform)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        features, labels = next(iter(loader))

    assert features.shape == (2, 38, 2, 30)
    assert labels.shape == (2, 38, 2)
    assert not torch.isnan(features).any()
    assert labels.max() <= 2
    assert labels.min() >= 0


if __name__ == "__main__":
    test_pipeline()
