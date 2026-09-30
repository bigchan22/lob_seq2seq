import torch
import torch.nn as nn
import math

class LOBTransformer(nn.Module):
    def __init__(self, num_assets, num_features, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_assets = num_assets
        
        # 1. Input Projection: Flatten N assets * F features into d_model
        # This allows the model to learn cross-asset correlations immediately
        self.input_projection = nn.Linear(num_assets * num_features, d_model)
        
        # 2. Positional Encoding (Learnable)
        # Sequence length is 38
        self.pos_embedding = nn.Parameter(torch.zeros(1, 38, d_model))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Output Head
        # Projects back to (N assets * 3 classes)
        self.output_head = nn.Linear(d_model, num_assets * 3)
        
        self.dropout = nn.Dropout(dropout)

    def generate_causal_mask(self, sz, device):
        """Generates a square mask for the sequence. The masked positions are filled with float('-inf')."""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        # x shape: (B, 38, N, F)
        batch_size, seq_len, N, F = x.shape
        
        # Flatten Assets and Features: (B, 38, N*F)
        x = x.view(batch_size, seq_len, -1)
        
        # Project to d_model and add Positional Encoding
        x = self.input_projection(x) # (B, 38, d_model)
        x = x + self.pos_embedding[:, :seq_len]
        x = self.dropout(x)
        
        # Generate and apply causal mask
        mask = self.generate_causal_mask(seq_len, x.device)
        
        # Transformer pass
        # The mask ensures at step t, the model only attends to 0...t
        output = self.transformer_encoder(x, mask=mask) # (B, 38, d_model)
        
        # Project to logits
        logits = self.output_head(output) # (B, 38, N*3)
        
        # Reshape back to (B, 38, N, 3) for the 3 classes (Down, Flat, Up)
        logits = logits.view(batch_size, seq_len, N, 3)
        
        return logits


class AssetAwareLOBTransformer(nn.Module):
    def __init__(
        self,
        num_assets,
        num_features,
        d_model=256,
        nhead=8,
        temporal_layers=4,
        cross_asset_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_assets = num_assets

        self.input_projection = nn.Linear(num_features, d_model)
        self.asset_embedding = nn.Parameter(torch.zeros(1, 1, num_assets, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, 38, 1, d_model))

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=temporal_layers)

        cross_asset_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_asset_encoder = nn.TransformerEncoder(
            cross_asset_layer,
            num_layers=cross_asset_layers,
        )

        self.output_head = nn.Linear(d_model, 3)
        self.dropout = nn.Dropout(dropout)

    def generate_causal_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))

    def forward(self, x):
        # x shape: (B, T, N, F)
        batch_size, seq_len, num_assets, _ = x.shape

        x = self.input_projection(x)
        x = x + self.pos_embedding[:, :seq_len] + self.asset_embedding[:, :, :num_assets]
        x = self.dropout(x)

        # Encode each asset's time history independently with a causal mask.
        temporal_input = x.permute(0, 2, 1, 3).reshape(batch_size * num_assets, seq_len, self.d_model)
        temporal_mask = self.generate_causal_mask(seq_len, x.device)
        temporal_output = self.temporal_encoder(temporal_input, mask=temporal_mask)
        temporal_output = temporal_output.reshape(batch_size, num_assets, seq_len, self.d_model).permute(0, 2, 1, 3)

        # Let assets exchange information at each time step after temporal encoding.
        cross_input = temporal_output.reshape(batch_size * seq_len, num_assets, self.d_model)
        cross_output = self.cross_asset_encoder(cross_input)
        cross_output = cross_output.reshape(batch_size, seq_len, num_assets, self.d_model)

        return self.output_head(cross_output)
