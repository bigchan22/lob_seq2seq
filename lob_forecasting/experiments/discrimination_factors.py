"""Versioned, target-excluding market factors for the Stage-2 development study."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

NAMES=('market_return_mean','market_relative_spread_mean','market_OBI1_mean',
       'market_OBI5_mean','market_fraction_positive','market_log_depth_mean')

def raw_asset_factors(x,eps=1e-9):
    """Recover financially interpretable quantities from the 13 no-investor features."""
    mid=torch.exp(x[...,0]);ret=torch.zeros_like(mid)
    ret[:,1:]=torch.log((mid[:,1:]+eps)/(mid[:,:-1]+eps)) # never crosses a day
    relative_spread=torch.exp(x[...,1])-torch.exp(x[...,2])
    asks=torch.stack([torch.expm1(x[...,3+2*k]).clamp_min(0) for k in range(5)],-1)
    bids=torch.stack([torch.expm1(x[...,4+2*k]).clamp_min(0) for k in range(5)],-1)
    def obi(levels):
        a=asks[...,:levels].sum(-1);b=bids[...,:levels].sum(-1)
        return (b-a)/(b+a+eps)
    depth=torch.log1p((asks+bids).sum(-1))
    return torch.stack((ret,relative_spread,obi(1),obi(5),(ret>0).to(x.dtype),depth),-1)

def leave_one_out(x,depth=True):
    raw=raw_asset_factors(x);n=raw.shape[2]
    if n<2: raise ValueError('leave-one-out factors require at least two assets')
    loo=(raw.sum(2,keepdim=True)-raw)/(n-1)
    return loo if depth else loo[...,:5]

@dataclass
class TrainFactorScaler:
    low: np.ndarray|None=None; high: np.ndarray|None=None
    mean: np.ndarray|None=None; std: np.ndarray|None=None
    def fit(self,v):
        a=np.asarray(v,dtype=np.float64);axes=tuple(range(a.ndim-1))
        self.low=np.quantile(a,.001,axis=axes);self.high=np.quantile(a,.999,axis=axes)
        c=np.clip(a,self.low,self.high);self.mean=c.mean(axis=axes);self.std=c.std(axis=axes)
        self.std=np.where(self.std<1e-12,1.,self.std);return self
    def transform(self,v):
        a=np.asarray(v,dtype=np.float64);return ((np.clip(a,self.low,self.high)-self.mean)/self.std).astype(np.float32)
    def state(self):return {k:getattr(self,k).tolist() for k in ('low','high','mean','std')}

def scaled_factors(x,train_end,depth=True):
    raw=leave_one_out(x,depth).cpu().numpy();scaler=TrainFactorScaler().fit(raw[:train_end])
    out=torch.from_numpy(scaler.transform(raw))
    if not torch.isfinite(out).all():raise ValueError('non-finite UM-v2 factor')
    return out,raw,scaler
