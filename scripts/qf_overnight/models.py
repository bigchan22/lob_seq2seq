"""Recovered architectures, GRU, common-query control and exact state interventions."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from torch import nn
from lob_forecasting.models.qf_variants import build_qf_model, CrossAssetAttentionModel

class GRUOwn(nn.Module):
    def __init__(self,n=27):
        super().__init__();self.input_projection=nn.Linear(13,128)
        self.asset_embedding=nn.Parameter(torch.zeros(1,n,128))
        self.gru=nn.GRU(128,128,2,batch_first=True,dropout=.1,bidirectional=False)
        self.output_head=nn.Linear(128,3)
    def forward(self,x):
        b,t,n,_=x.shape;h=self.input_projection(x)+self.asset_embedding[:,None,:n]
        h,_=self.gru(h.permute(0,2,1,3).reshape(b*n,t,128))
        return self.output_head(h.reshape(b,n,t,128).permute(0,2,1,3))

def cross_layer(layer,residual,query,key,value,mask=None):
    # Original postnorm TransformerEncoderLayer, including self, unchanged except query/KV.
    a=layer.self_attn(query,key,value,attn_mask=mask,need_weights=False)[0]
    h=layer.norm1(residual+layer.dropout1(a))
    return layer.norm2(h+layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))))

class SharedQuery(CrossAssetAttentionModel):
    def forward(self,x):
        h=self.encode(x);b,t,n,d=h.shape;h=h.reshape(b*t,n,d)
        for layer in self.cross_asset_encoder.layers:
            q=h.mean(1,keepdim=True).expand_as(h)
            h=cross_layer(layer,h,q,h,h)
        return self.output_head(h.reshape(b,t,n,d))

def build(model,n=27):
    if model=='GRU_U1':return GRUOwn(n)
    if model=='TLOB_ADAPTED':
        from tlob_adapter import TLOBAdapted
        return TLOBAdapted(n)
    kw=dict(num_assets=n,num_features=13,d_model=128,nhead=8,temporal_layers=2,
            cross_asset_layers=1,dropout=.1)
    return SharedQuery(**kw) if model=='SHARED_QUERY' else build_qf_model(model,**kw)

def forward(model,name,x,factors=None):
    return model(x,factors) if name=='UM_V2_DEPTH' else model(x)

def ua_intervention(model,x,condition):
    h=model.encode(x);b,t,n,d=h.shape
    assert len(model.cross_asset_encoder.layers)==1
    layer=model.cross_asset_encoder.layers[0]
    if condition=='self_only':
        flat=h.reshape(b*t,n,d);mask=~torch.eye(n,dtype=torch.bool,device=x.device)
        return model.output_head(cross_layer(layer,flat,flat,flat,flat,mask).reshape(b,t,n,d))
    if condition=='peer_lag1':
        previous=torch.cat([h[:,:1],h[:,:-1]],1);result=[]
        # Target state and own key/value remain current; other keys/values are t-1.
        # Temporal states were computed on original full history with original positions.
        for target in range(n):
            kv=previous.clone();kv[:,:,target]=h[:,:,target]
            query=h[:,:,target:target+1].reshape(b*t,1,d)
            out=cross_layer(layer,query,query,kv.reshape(b*t,n,d),kv.reshape(b*t,n,d))
            result.append(out.reshape(b,t,1,d))
        return model.output_head(torch.cat(result,2))
    raise ValueError(condition)
