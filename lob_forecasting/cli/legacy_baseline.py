import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from lob_forecasting.data.dataset import LOBDataset
from lob_forecasting.data.transforms import LOBFeatureTransformer
from lob_forecasting.data.preprocess import get_ticker_list

# --- 1. Baseline Architectures ---

# class MLPBaseline(nn.Module):
#     """Deep but non-sequential baseline."""
#     def __init__(self, num_assets, num_features, seq_len=38):
#         super().__init__()
#         self.input_dim = seq_len * num_assets * num_features
#         self.net = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(self.input_dim, 1024),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, num_assets * 3) 
#         )

#     def forward(self, x):
#         # x: (Batch, Time, Assets, Features)
        
#         batch_size = x.size(0)
#         logits = self.net(x)
#         # 다시 (Batch, Time, Assets, 3) 형태로 복원하여 Transformer와 형식을 맞춤
#         return logits.view(batch_size, self.seq_len, self.num_assets, 3)
        
#         # logits = self.net(x)
#         # return logits.view(x.size(0), x.size(2), 3) # (Batch, Assets, 3)
class MLPBaseline(nn.Module):
    def __init__(self, num_assets, num_features, seq_len=38):
        super().__init__()
        # self.를 붙여서 클래스 전체에서 사용할 수 있게 저장합니다.
        self.num_assets = num_assets
        self.seq_len = seq_len
        
        self.input_dim = seq_len * num_assets * num_features
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            # 출력 크기: (Batch, Time * Assets * 3)
            nn.Linear(512, self.seq_len * self.num_assets * 3) 
        )

    def forward(self, x):
        batch_size = x.size(0)
        logits = self.net(x)
        # 여기서 self.seq_len과 self.num_assets를 사용하여 차원을 복구합니다.
        return logits.view(batch_size, self.seq_len, self.num_assets, 3)
class LSTMBaseline(nn.Module):
    def __init__(self, num_assets, num_features, seq_len=38, hidden_dim=256, num_layers=2):
        super().__init__()
        self.num_assets = num_assets
        self.seq_len = seq_len
        
        # LSTM 입력 차원: 모든 종목의 피처를 합친 크기
        self.lstm = nn.LSTM(
            input_size=num_assets * num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 출력: (Batch, Time, hidden_dim) -> (Batch, Time, Assets * 3)
        self.fc = nn.Linear(hidden_dim, num_assets * 3)

    def forward(self, x):
        b, t, n, f = x.shape
        # Assets와 Features를 하나로 묶어 LSTM 입력으로 변환: (Batch, Time, N*F)
        x = x.view(b, t, -1)
        
        # out shape: (Batch, Time, hidden_dim)
        out, _ = self.lstm(x)
        
        # 최종 클래스 분류
        logits = self.fc(out)
        
        # Transformer와 동일한 형태로 복원: (Batch, Time, Assets, 3)
        return logits.view(b, t, n, 3)
        
class LinearBaseline(nn.Module):
    """Essentially Multi-output Logistic Regression."""
    def __init__(self, num_assets, num_features, seq_len=38):
        super().__init__()
        self.input_dim = seq_len * num_assets * num_features
        self.linear = nn.Linear(self.input_dim, num_assets * 3)

    def forward(self, x):
        logits = self.linear(x.view(x.size(0), -1))
        return logits.view(x.size(0), x.size(2), 3)

# --- 2. Training Function ---

def train_baseline(model_type="mlp"):
    # --- 1. Configuration ---
    DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    DATA_DIR = "./data/processed"
    BATCH_SIZE = 64
    EPOCHS = 500
    LEARNING_RATE = 1e-4
    EPSILON = 0.0001
    SPLIT_RATIO = 0.8
    include_investor= True
    tickers = get_ticker_list(DATA_DIR)
    
    # --- 이 부분을 확인하세요 ---
    num_assets = len(tickers)
    
    
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

    
    # --- Initialize Model ---
    input_dim = full_x.shape[-1]
    # if model_type == "mlp":
    #     model = MLPBaseline(len(tickers), input_dim).to(DEVICE)
    # else:
    #     model = LinearBaseline(len(tickers), input_dim).to(DEVICE)
    if model_type == "mlp":
        model = MLPBaseline(num_assets, input_dim).to(DEVICE)
    elif model_type == "lstm":
        model = LSTMBaseline(num_assets, input_dim).to(DEVICE)
    elif model_type == "linear":
        # 위에서 정의한 LinearBaseline 사용 시
        model = LinearBaseline(num_assets, input_dim).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    # Use the same loop from your main.py
    # Note: In logits = model(features), ensure you use logits.view(-1, 3)
    # just like in the Transformer version for a fair comparison.
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
            torch.save(model.state_dict(), f"best_lob_model_baseline_{model_type}_{include_investor}.pth")
            print(">>> Best f1 model saved!")
        print("-" * 30)

if __name__ == "__main__":
    train_baseline()
