from pathlib import Path
def csv_tickers(data_dir): return sorted(p.stem for p in Path(data_dir).glob("*.csv"))
