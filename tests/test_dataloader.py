import torch
from torch.utils.data import DataLoader
from src.data_loader.dataset import LOBDataset
from src.data_loader.transforms import LOBNormalizer

def test_pipeline():
    # 1. Setup parameters
    data_path = "./data/raw/KRXaggData"
    # Select a subset of tickers you actually have files for
    selected_tickers = ["TICKER1", "TICKER2", "TICKER3"] 
    batch_size = 4
    num_assets = len(selected_tickers)
    num_features = 8 # Update this based on your actual feature_cols count
    
    # 2. Instantiate Dataset with Normalization
    # Note: We'll assume you added price/qty indices to your normalizer
    norm = LOBNormalizer(price_indices=[0, 1, 4, 5], qty_indices=[2, 3, 6, 7])
    dataset = LOBDataset(data_dir=data_path, ticker_list=selected_tickers, transform=norm)
    
    # 3. Instantiate DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 4. Fetch one batch
    features, labels = next(iter(loader))
    
    # --- Assertions ---
    print("Checking Tensor Shapes...")
    # Expected: (B, 39, N, F)
    assert features.shape == (batch_size, 39, num_assets, num_features), \
        f"Feature shape mismatch: {features.shape}"
        
    # Expected: (B, 39, N)
    assert labels.shape == (batch_size, 39, num_assets), \
        f"Label shape mismatch: {labels.shape}"
    
    print("Checking Data Integrity...")
    # Check for NaNs (common in LOB data due to zero-division in spreads)
    assert not torch.isnan(features).any(), "Found NaNs in features!"
    
    # Check label range (0: Down, 1: Flat, 2: Up)
    assert labels.max() <= 2 and labels.min() >= 0, "Labels out of bounds!"

    print("\n✅ Pipeline Test Passed!")
    print(f"Batch Features Shape: {features.shape}")
    print(f"Batch Labels Shape:   {labels.shape}")

if __name__ == "__main__":
    test_pipeline()
