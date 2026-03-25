import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
def get_valid_intervals():
    """Generates the 39 required intervals: 09:00, 09:10 ... 15:20"""
    # Adjust format logic based on how your TIME_INTERVAL is stored (e.g., 900 or "09:00")
    # Assuming it's an integer like 900, 910, etc.
    intervals = []
    # 09:00 to 15:20
    for h in range(9, 16):
        for m in range(0, 60, 10):
            time_val = h * 100 + m
            if time_val > 1520: break
            intervals.append(time_val)
    return intervals
def get_ticker_list(data_dir):
    """
    Scans the directory for CSV files and returns a list of ticker names 
    without the .csv extension.
    """
    data_path = Path(data_dir)
    # Get all .csv files and strip the extension
    tickers = [f.stem for f in data_path.glob("*.csv")]
    
    if not tickers:
        print(f"⚠️ Warning: No CSV files found in {data_dir}")
        
    return sorted(tickers)
def convert_and_detect_files(raw_dir, processed_dir):
    """
    Scans for .xlsx files, converts them to .csv for speed, 
    and returns the list of processed filenames.
    """
    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Automatically detect all .xlsx files
    xlsx_files = list(raw_path.glob("*.xlsx"))
    print(f"Found {len(xlsx_files)} Excel files. Starting conversion...")
    
    ticker_list = []
    
    for file in tqdm(xlsx_files):
        csv_name = file.stem + ".csv"  # e.g., KR7000030007_t.csv
        target_path = proc_path / csv_name
        
        # 2. Conversion (only if CSV doesn't exist to save time)
        if not target_path.exists():
            df = pd.read_excel(file)
            df.to_csv(target_path, index=False)
        
        ticker_list.append(csv_name)
        
    return ticker_list

# Usage
# tickers = convert_and_detect_files('./data/raw/KRXaggData', './data/processed/')