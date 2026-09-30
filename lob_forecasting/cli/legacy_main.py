import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm  # Progress bar

from lob_forecasting.data.dataset import LOBDataset
from lob_forecasting.data.transforms import LOBFeatureTransformer
from lob_forecasting.models.transformers import LOBTransformer
from lob_forecasting.data.preprocess import get_ticker_list

def train():
    # --- 1. Configuration ---
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DATA_DIR = "./data/processed"
    BATCH_SIZE = 64
    EPOCHS = 500
    LEARNING_RATE = 1e-4
    EPSILON = 0.0001
    SPLIT_RATIO = 0.8
    include_investor= False
    d_model=256
    nhead=16
    num_layers=4


    
# --- 2. Data Setup (Pre-load to VRAM) ---
    tickers = get_ticker_list(DATA_DIR)
    temp_ds = LOBDataset(DATA_DIR, tickers, epsilon=EPSILON)
    feature_transformer = LOBFeatureTransformer(col_map=temp_ds.col_map, include_investor=include_investor)
    
    # This is the original dataset that reads CSVs
    raw_dataset = LOBDataset(DATA_DIR, tickers, epsilon=EPSILON, transform=feature_transformer)
    
    print(f"Pre-loading data to {DEVICE}... with including investor as {include_investor}")
    all_x, all_y = [], []
    for i in tqdm(range(len(raw_dataset)), desc="Baking Tensors"):
        x, y = raw_dataset[i]
        all_x.append(x)
        all_y.append(y)

    # Convert lists to single giant tensors and move to GPU immediately
    full_x = torch.stack(all_x).to(DEVICE)           # Shape: (Days, 38, N, F)
    full_y = torch.stack(all_y).to(DEVICE).long()    # Shape: (Days, 38, N)
    
    # Chronological Split
    train_size = int(len(full_x) * SPLIT_RATIO)
    
    # Use TensorDataset: it's designed for data already in memory
    from torch.utils.data import TensorDataset
    train_ds = TensorDataset(full_x[:train_size], full_y[:train_size])
    test_ds = TensorDataset(full_x[train_size:], full_y[train_size:])

    # Note: num_workers=0 because data is already on GPU
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # --- 3. Model ---
    # sample_x, _ = full_dataset[0]
    input_dim = full_x.shape[-1]
    model = LOBTransformer(
        num_assets=len(tickers),
        num_features=input_dim,
        d_model=d_model, nhead=nhead, num_layers=num_layers
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 4. Training Loop with tqdm ---
    best_f1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        # Wrap the loader in tqdm
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for features, labels in pbar:
            # features, labels = features.to(DEVICE), labels.to(DEVICE).long()
            
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits.view(-1, 3), labels.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            # Update the progress bar with current loss
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # --- 5. Evaluation Phase ---
        model.eval()
        all_preds, all_labels = [], []
        
        print(f"\nEvaluating Epoch {epoch+1}...")
        with torch.no_grad():
            for features, labels in tqdm(test_loader, desc="Testing"):
                # features = features.to(DEVICE)
                logits = model(features)
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.append(preds.view(-1))
                all_labels.append(labels.view(-1))

# Flatten the list of tensors into one tensor, move to CPU, then NumPy
        y_true = torch.cat(all_labels).cpu().numpy()
        y_pred = torch.cat(all_preds).cpu().numpy()
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')

        print(f"\n[Epoch {epoch+1} Results]")
        print(f"Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")
        
        # Detail: Check if model is actually predicting all classes
        unique, counts = np.unique(y_pred, return_counts=True)
        print(f"Pred Distribution: {dict(zip(unique, counts))}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f"best_lob_mode_{include_investor}_{d_model}_{nhead}_{num_layers}.pth")
            print(">>> Best f1 model saved!")
        print("-" * 30)

if __name__ == "__main__":
    train()