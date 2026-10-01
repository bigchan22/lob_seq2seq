"""Official TLOB dual-attention backbone on own-asset causal aggregate prefixes.

Fixed 38-slot daily-origin window. For origin t, slots 0..t contain the original
same-day history; slots after t repeat the current vector, never future inputs.
The unmasked upstream backbone is bidirectional only within that known prefix
and its deterministic padding. No peers, asset embedding or event features.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from vendor_tlob.tlob import TLOB

class TLOBAdapted(nn.Module):
    def __init__(self,n=27):
        super().__init__();self.core=TLOB(hidden_dim=40,num_layers=4,seq_size=38,
             num_features=13,num_heads=1,is_sin_emb=True,dataset_type='KRX_AGGREGATE_PREFIX')
    def forward(self,x):
        b,t,n,f=x.shape;assert f==13 and t<=38
        outputs=[];positions=torch.arange(38,device=x.device)
        for origin in range(t):
            prefix=x[:,torch.minimum(positions,torch.tensor(origin,device=x.device))]
            prefix=prefix.permute(0,2,1,3).reshape(b*n,38,13)
            # Exact activation recomputation, no mixed precision or model change.
            # Chunking changes memory only; normalization is per sample.
            chunks=[]
            for start in range(0,len(prefix),128):
                part=prefix[start:start+128]
                if self.training and torch.is_grad_enabled():
                    if not part.requires_grad:part=part.detach().requires_grad_(True)
                    chunks.append(checkpoint(self.core,part))
                else:chunks.append(self.core(part))
            outputs.append(torch.cat(chunks).reshape(b,n,3))
        return torch.stack(outputs,1)
