"""Controlled QF decomposition models; historical model classes are untouched."""
from __future__ import annotations
import torch
from torch import nn

def causal_mask(length, device):
    return torch.triu(torch.full((length, length), float("-inf"), device=device), diagonal=1)

class UniversalTemporalModel(nn.Module):
    """U0/U1: shared causal encoder applied independently to every asset."""
    def __init__(self, num_assets, num_features, d_model=128, nhead=8, temporal_layers=2,
                 dropout=.1, asset_embedding=False, extra_own_layers=0):
        super().__init__(); self.num_assets=num_assets; self.d_model=d_model
        self.input_projection=nn.Linear(num_features,d_model)
        self.pos_embedding=nn.Parameter(torch.zeros(1,38,d_model))
        self.asset_embedding=nn.Parameter(torch.zeros(1,num_assets,d_model)) if asset_embedding else None
        layer=lambda: nn.TransformerEncoderLayer(d_model,nhead,d_model*4,dropout,batch_first=True,norm_first=False)
        self.temporal_encoder=nn.TransformerEncoder(layer(),temporal_layers)
        self.extra_own_encoder=nn.TransformerEncoder(layer(),extra_own_layers) if extra_own_layers else None
        self.output_head=nn.Linear(d_model,3);self.dropout=nn.Dropout(dropout)
    def encode(self,x):
        b,t,n,_=x.shape;h=self.input_projection(x).permute(0,2,1,3).reshape(b*n,t,self.d_model)
        h=h+self.pos_embedding[:,:t]
        if self.asset_embedding is not None:h=h+self.asset_embedding[:,:n].reshape(n,1,self.d_model).repeat(b,1,1)
        h=self.temporal_encoder(self.dropout(h),mask=causal_mask(t,x.device))
        if self.extra_own_encoder is not None:h=self.extra_own_encoder(h,mask=causal_mask(t,x.device))
        return h.reshape(b,n,t,self.d_model).permute(0,2,1,3)
    def forward(self,x):return self.output_head(self.encode(x))

class MarketAugmentedModel(UniversalTemporalModel):
    """UM: U1 plus target-excluding aggregate factors, never individual peer features."""
    def __init__(self,num_assets,num_features,num_market_factors=6,**kw):
        super().__init__(num_assets,num_features,asset_embedding=True,**kw)
        self.market_projection=nn.Linear(num_market_factors,self.d_model)
    def forward(self,x,market_factors):return self.output_head(self.encode(x)+self.market_projection(market_factors))

class StandardizedMarketModel(UniversalTemporalModel):
    """UM-v2: U1 plus a small branch for train-standardized LOO factors."""
    def __init__(self,num_assets,num_features,num_market_factors=5,**kw):
        super().__init__(num_assets,num_features,asset_embedding=True,**kw)
        self.factor_branch=nn.Sequential(nn.LayerNorm(num_market_factors),
            nn.Linear(num_market_factors,self.d_model),nn.GELU(),nn.Dropout(kw.get('dropout',.1)),
            nn.Linear(self.d_model,self.d_model))
    def forward(self,x,market_factors):
        return self.output_head(self.encode(x)+self.factor_branch(market_factors))

class StaticMixerModel(UniversalTemporalModel):
    """UX: learned content-independent same-time linear map over assets."""
    def __init__(self,num_assets,num_features,**kw):
        super().__init__(num_assets,num_features,asset_embedding=True,**kw)
        self.asset_mixer=nn.Parameter(torch.eye(num_assets))
    def forward(self,x):
        h=self.encode(x);mixed=torch.einsum("ij,btjd->btid",self.asset_mixer,h)
        return self.output_head(h+mixed)

class CapacityControlModel(UniversalTemporalModel):
    """UC: own-history-only extra causal encoder exactly matches UA cross-layer capacity."""
    def __init__(self,num_assets,num_features,cross_asset_layers=1,**kw):
        super().__init__(num_assets,num_features,asset_embedding=True,extra_own_layers=cross_asset_layers,**kw)

class CrossAssetAttentionModel(UniversalTemporalModel):
    """UA: U1 plus content-dependent attention among assets at the same timestamp."""
    def __init__(self,num_assets,num_features,cross_asset_layers=1,**kw):
        super().__init__(num_assets,num_features,asset_embedding=True,**kw)
        layer=nn.TransformerEncoderLayer(self.d_model,kw.get("nhead",8),self.d_model*4,kw.get("dropout",.1),batch_first=True,norm_first=False)
        self.cross_asset_encoder=nn.TransformerEncoder(layer,cross_asset_layers)
    def forward(self,x):
        h=self.encode(x);b,t,n,d=h.shape
        h=self.cross_asset_encoder(h.reshape(b*t,n,d)).reshape(b,t,n,d)
        return self.output_head(h)

def build_qf_model(model_id, **kw):
    table={"S":UniversalTemporalModel,"U0":UniversalTemporalModel,"U1":UniversalTemporalModel,
           "UM":MarketAugmentedModel,"UX":StaticMixerModel,"UC":CapacityControlModel,"UA":CrossAssetAttentionModel}
    if model_id in {'UM_V2_LITE','UM_V2_DEPTH'}:
        kw.pop('cross_asset_layers',None);kw['num_market_factors']=5 if model_id.endswith('LITE') else 6
        return StandardizedMarketModel(**kw)
    if model_id not in table:raise ValueError(model_id)
    if model_id not in {"UC","UA"}:kw.pop("cross_asset_layers",None)
    if model_id in {"U1"}:kw["asset_embedding"]=True
    if model_id=="S":kw["num_assets"]=1
    return table[model_id](**kw)

def trainable_parameters(model):return sum(p.numel() for p in model.parameters() if p.requires_grad)
