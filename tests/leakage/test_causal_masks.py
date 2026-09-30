import pytest
torch=pytest.importorskip("torch")
from lob_forecasting.models.transformers import AssetAwareLOBTransformer,LOBTransformer

@pytest.mark.parametrize("aware",[False,True])
def test_future_cannot_change_earlier_logits(aware):
 torch.manual_seed(1); x=torch.randn(1,38,3,4)
 m=(AssetAwareLOBTransformer(3,4,d_model=16,nhead=4,temporal_layers=1,cross_asset_layers=1,dropout=0) if aware else LOBTransformer(3,4,d_model=16,nhead=4,num_layers=1,dropout=0)).eval()
 with torch.no_grad(): a=m(x); z=x.clone(); z[:,20:]+=100; b=m(z)
 assert torch.max(torch.abs(a[:,:20]-b[:,:20])).item()<1e-7

def test_same_timestamp_other_asset_can_affect_target():
 torch.manual_seed(2);x=torch.randn(1,38,3,4);m=AssetAwareLOBTransformer(3,4,d_model=16,nhead=4,temporal_layers=1,cross_asset_layers=1,dropout=0).eval()
 with torch.no_grad():a=m(x);z=x.clone();z[:,10,1]+=10;b=m(z)
 assert not torch.equal(a[:,10,0],b[:,10,0])
