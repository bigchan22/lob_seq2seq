import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class LOBDataset(Dataset):
    def __init__(self, data_dir, ticker_list, epsilon=0.0001, transform=None):
        self.data_dir = Path(data_dir)
        self.tickers = ticker_list
        self.transform = transform
        self.epsilon = epsilon
        
        # 1. Align Dates (Indentation fixed)
        self.common_dates = self._get_common_dates()

        self.master_grid = [
            "09:00:00", "09:10:00", "09:20:00", "09:30:00", "09:40:00", "09:50:00",
            "10:00:00", "10:10:00", "10:20:00", "10:30:00", "10:40:00", "10:50:00",
            "11:00:00", "11:10:00", "11:20:00", "11:30:00", "11:40:00", "11:50:00",
            "12:00:00", "12:10:00", "12:20:00", "12:30:00", "12:40:00", "12:50:00",
            "13:00:00", "13:10:00", "13:20:00", "13:30:00", "13:40:00", "13:50:00",
            "14:00:00", "14:10:00", "14:20:00", "14:30:00", "14:40:00", "14:50:00",
            "15:00:00", "15:10:00", "15:20:00"
        ]
        
        # Define all columns to be read from CSV
        self.raw_cols = [
            'SELL_PRC11', 'BUY_PRC11', 'SELL_PRC12', 'BUY_PRC12',
            'SELL_PRC21', 'BUY_PRC21', 'SELL_PRC22', 'BUY_PRC22',
            'SELL_QTY11', 'BUY_QTY11', 'SELL_QTY12', 'BUY_QTY12',
            'SELL_QTY21', 'BUY_QTY21', 'SELL_QTY22', 'BUY_QTY22',
            'TRD_PRC', 
            'ASK_STEP1_BSTORD_PRC', 'ASK_STEP1_BSTORD_RQTY',
            'ASK_STEP2_BSTORD_PRC', 'ASK_STEP2_BSTORD_RQTY',
            'ASK_STEP3_BSTORD_PRC', 'ASK_STEP3_BSTORD_RQTY',
            'ASK_STEP4_BSTORD_PRC', 'ASK_STEP4_BSTORD_RQTY',
            'ASK_STEP5_BSTORD_PRC', 'ASK_STEP5_BSTORD_RQTY',
            'BID_STEP1_BSTORD_PRC', 'BID_STEP1_BSTORD_RQTY',
            'BID_STEP2_BSTORD_PRC', 'BID_STEP2_BSTORD_RQTY',
            'BID_STEP3_BSTORD_PRC', 'BID_STEP3_BSTORD_RQTY',
            'BID_STEP4_BSTORD_PRC', 'BID_STEP4_BSTORD_RQTY',
            'BID_STEP5_BSTORD_PRC', 'BID_STEP5_BSTORD_RQTY',
            'SELL_PRC2_AGG_WAVG', 'BUY_PRC2_AGG_WAVG',
            'SELL_QTY2_AGG_SUM', 'BUY_QTY2_AGG_SUM'
        ]
                
        # Column map for the Transformer to know which index is what
        self.col_map = {name: i for i, name in enumerate(self.raw_cols)}

    def _get_common_dates(self):
        date_sets = []
        for ticker in self.tickers:
            df = pd.read_csv(self.data_dir / f"{ticker}.csv", usecols=['ORD_DD'])
            date_sets.append(set(df['ORD_DD'].unique()))
        return sorted(list(set.intersection(*date_sets)))

    def __len__(self):
        return len(self.common_dates)

    def __getitem__(self, idx):
        target_date = self.common_dates[idx]
        day_features = []
        day_labels = []
        skeleton = pd.DataFrame({'TIME_INTERVAL': self.master_grid})

        for ticker in self.tickers:
            df = pd.read_csv(self.data_dir / f"{ticker}.csv")
            df_day = df[df['ORD_DD'] == target_date].copy()
            df_day['TIME_INTERVAL'] = df_day['TIME_INTERVAL'].astype(str).str.strip()

            df_aligned = pd.merge(skeleton, df_day, on='TIME_INTERVAL', how='left')
            
            # 1. Fill gaps without looking ahead.  Backward-fill would copy
            # later same-day values into earlier intervals and leak future LOB
            # information into the prediction input.
            df_aligned = df_aligned.ffill().fillna(0)

            # 2. Slice Features (Indices 0 to 37)
            x_raw = df_aligned[self.raw_cols].values[:-1, :].astype(np.float32)

            # 3. Safe Label Calculation
            # Check if columns exist to avoid KeyError
            if 'ASK_STEP1_BSTORD_PRC' in df_aligned.columns and 'BID_STEP1_BSTORD_PRC' in df_aligned.columns:
                midprices = (df_aligned['ASK_STEP1_BSTORD_PRC'].values + 
                             df_aligned['BID_STEP1_BSTORD_PRC'].values) / 2
            else:
                # Fallback if columns are missing
                midprices = np.zeros(len(df_aligned))
            
            ticker_labels = []
            for t in range(len(midprices) - 1):
                m_t = midprices[t]
                m_next = midprices[t+1]
                
                # Logic: Only calculate diff if current price is valid (non-zero)
                if m_t > 0:
                    diff = (m_next - m_t) / m_t
                    if diff > self.epsilon: 
                        ticker_labels.append(2) # Up
                    elif diff < -self.epsilon: 
                        ticker_labels.append(0) # Down
                    else: 
                        ticker_labels.append(1) # Flat
                else:
                    # If midprice is 0 (halted/no data), we cannot predict a move
                    ticker_labels.append(1) # Default to Flat
            
            day_features.append(x_raw)
            day_labels.append(ticker_labels)

        # 4. Shape Transformation (38, N, F)
        x = torch.from_numpy(np.stack(day_features)).transpose(0, 1)
        y = torch.from_numpy(np.stack(day_labels)).transpose(0, 1)

        # Final safety check: replace any NaNs in features with 0 before passing to model
        x = torch.nan_to_num(x, nan=0.0)

        if self.transform:
            x = self.transform(x)

        return x, y
