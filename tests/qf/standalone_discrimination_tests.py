"""Executable Stage-2 integrity suite for environments without pytest."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from lob_forecasting.experiments.discrimination_factors import raw_asset_factors,leave_one_out,scaled_factors
from lob_forecasting.experiments.discrimination_analysis import align,boot
from lob_forecasting.experiments.discrimination_plan import build_plan
from lob_forecasting.models.qf_variants import build_qf_model,trainable_parameters
def check(name,fn):fn();print('PASS',name)
def factors():
 x=torch.randn(10,8,27,13);x[...,3:]=x[...,3:].abs();raw=raw_asset_factors(x);assert torch.all(raw[:,0,:,0]==0);z,_,_=scaled_factors(x,7,True);assert z.shape[-1]==6 and torch.isfinite(z).all();assert abs(float(z[:7].mean()))<.1
def alignment():
 d=pd.DataFrame({'date':['d'],'timestamp':['t'],'asset':['a'],'split':['validation'],'seed':[1],'horizon':[1],'protocol_id':['p'],'feature_set':['no_investor'],'true_class':[0]});align(d,d.copy())
def bootstrap():assert np.array_equal(boot(np.arange(10),20,4),boot(np.arange(10),20,4))
def models():
 kw=dict(num_assets=27,num_features=13,d_model=32,nhead=4,temporal_layers=1,cross_asset_layers=1,dropout=0.)
 ua=build_qf_model('UA',**kw);uc=build_qf_model('UC',**kw);assert abs(trainable_parameters(ua)-trainable_parameters(uc))/trainable_parameters(ua)<.05
 x=torch.randn(2,6,27,13);y=ua(x);x[:,5]+=100;assert torch.allclose(y[:,:5],ua(x)[:,:5],atol=1e-6)
def plan():
 p=build_plan()['jobs'];assert not any('lockbox' in str(x).lower() or 'rolling' in str(x).lower() for x in p);assert sum(x['model']=='UA' and 'PAIR_' in x['job_id'] for x in p)==5;assert sum(x['model']=='UC' and 'PAIR_' in x['job_id'] for x in p)==5
def gpu():
 from lob_forecasting.experiments.queue_gates import validate_gpu
 try:validate_gpu(2);raise AssertionError
 except ValueError:pass
if __name__=='__main__':
 for n,f in [('market factors',factors),('alignment',alignment),('bootstrap',bootstrap),('models',models),('plan',plan),('gpu policy',gpu)]:check(n,f)
