from torch import nn

class MLP(nn.Module):
    def __init__(self, 
                 start_dim: int,
                 hidden_dim: int,
                 final_dim: int
                 ) -> None:
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(final_dim)
        self.fc = nn.Linear(start_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, final_dim)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        residual = x
        x = self.fc(x)
        x = self.gelu(x)
        x = self.fc2(x)
        if x.shape[2] == residual.shape[2]:
            x = x + residual
        x = self.layer_norm(x)
        x = self.gelu(x)
        return x
    
