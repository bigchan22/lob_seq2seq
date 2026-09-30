"""Point-in-time, leave-one-target-out market factors for UM."""
import torch

FACTOR_NAMES=("market_return_mean","market_relative_spread_mean","market_level1_imbalance_mean",
              "market_level5_imbalance_mean","market_depth_mean","fraction_positive_recent_return")

def leave_one_out_market_factors(mid,ask,bid,ask_qty,bid_qty,epsilon=1e-9):
    """Inputs [B,T,N] and quantities [B,T,N,L]; output [B,T,N,6]."""
    ret=torch.zeros_like(mid);ret[:,1:]=(mid[:,1:]-mid[:,:-1])/(mid[:,:-1].abs()+epsilon)
    spread=(ask-bid)/(mid.abs()+epsilon)
    def obi(l):
        b=bid_qty[...,:l].sum(-1);a=ask_qty[...,:l].sum(-1);return (b-a)/(b+a+epsilon)
    values=[ret,spread,obi(1),obi(min(5,ask_qty.shape[-1])),(bid_qty+ask_qty).sum(-1), (ret>0).to(mid.dtype)]
    n=mid.shape[2]
    if n<2:raise ValueError("leave-one-out factors require at least two assets")
    return torch.stack([(v.sum(2,keepdim=True)-v)/(n-1) for v in values],dim=-1)

def source_asset_counts(num_assets):
    if num_assets<2:raise ValueError("at least two assets required")
    return [num_assets-1]*num_assets
