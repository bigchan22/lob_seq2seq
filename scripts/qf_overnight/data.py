"""Exact frozen labels/masks and causal 13-feature inputs; task-local cache only."""
import csv, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import ROOT,POPS,SPLIT_INDICES,read,write,sha,digest
sys.path.insert(0,str(ROOT/'scripts/qf_data_review'))
from build_data_review import verify_inputs,align_quotes,ASK,BID,GRID
from lob_forecasting.data.transforms import LOBFeatureTransformer
from lob_forecasting.experiments.discrimination_factors import leave_one_out,TrainFactorScaler

def prepare(inputs,destination):
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'identity.json').exists():
        identity=read(dest/'identity.json')
        assert all(sha(dest/k)==v for k,v in identity['files'].items())
        return identity
    manifest=list(csv.DictReader((ROOT/'recovery/input_manifest.csv').open()))
    verified=verify_inputs(Path(inputs),manifest)
    assert len(verified)==54 and all(r['matches_pinned_manifest'] for r in verified)
    rows=pd.read_csv(ROOT/'artifacts/qf_data_review/20261001/forecast_rows.csv.gz',dtype={'date':str,'asset_id':str})
    assets=sorted(rows.asset_id.unique());dates=sorted(rows.date.unique())
    assert len(assets)==27 and len(dates)==493
    cols=[ASK,BID]+[side+'_STEP%d_BSTORD_RQTY'%i for i in range(1,6) for side in ['ASK','BID']]
    transformer=LOBFeatureTransformer({k:i for i,k in enumerate(cols)},include_investor=False)
    xs=[];ys=[];raw_cov=[]
    tokens=[d.replace('-','') for d in dates]
    for asset in assets:
        frame=pd.read_csv(Path(inputs)/'data/processed'/(asset+'.csv'),dtype={'ORD_DD':str,'TIME_INTERVAL':str})
        aligned,mid,labels,_=align_quotes(frame,tokens)
        filled=aligned[cols].groupby(level=0,sort=False).ffill().fillna(0)
        raw=filled.to_numpy(np.float32).reshape(493,39,len(cols))[:,:38]
        features=torch.nan_to_num(transformer(torch.from_numpy(raw))).numpy()
        assert features.shape==(493,38,13) and np.isfinite(features).all()
        xs.append(features);ys.append(labels)
        original=aligned[cols].to_numpy(np.float64).reshape(493,39,len(cols))[:,:38]
        a,b=original[...,0],original[...,1];m=(a+b)/2
        spread=np.full(m.shape,np.nan);good=np.isfinite(a)&np.isfinite(b)&(a>0)&(b>0)&(a>=b)
        spread[good]=(a[good]-b[good])/m[good]
        q=original[...,2:];depth=q.sum(-1);depth[~(np.isfinite(q).all(-1)&(q>=0).all(-1))]=np.nan
        raw_cov.append(np.stack([spread,depth],-1))
    x=np.stack(xs,2);y=np.stack(ys,2);cov=np.stack(raw_cov,2)
    index=pd.MultiIndex.from_product([dates,GRID[:-1],assets],names=['date','origin_time','asset_id'])
    aligned_rows=rows.set_index(['date','origin_time','asset_id']).reindex(index)
    assert np.array_equal(y.reshape(-1),aligned_rows.legacy_label.to_numpy())
    masks=aligned_rows[POPS].to_numpy(np.bool_).reshape(493,38,27,4)
    assert masks.sum((0,1,2)).tolist()==[505818,469391,439263,432560]
    expected=[[353970,323955,307395,302922],[50274,48280,43659,43050],[101574,97156,88209,86588]]
    for (a,b),count in zip([(0,345),(345,394),(394,493)],expected):assert masks[a:b].sum((0,1,2)).tolist()==count
    for name,value in [('x',x),('y',y),('masks',masks),('covariates',cov)]:
        with (dest/(name+'.npy')).open('xb') as f:np.save(f,value,allow_pickle=False)
    raw_factors=leave_one_out(torch.from_numpy(x),depth=True).numpy()
    with (dest/'raw_factors.npy').open('xb') as f:np.save(f,raw_factors,allow_pickle=False)
    identity={'input_manifest_sha256':sha(ROOT/'recovery/input_manifest.csv'),
              'handoff_sha':'e85a89fd56584b513be940e6259c637369bd830b',
              'forecast_rows_sha256':sha(ROOT/'artifacts/qf_data_review/20261001/forecast_rows.csv.gz'),
              'assets':assets,'dates':dates,'feature_policy':'legacy_no_investor_13',
              'input_files_verified':54,'label_mismatches':0,'mask_counts':masks.sum((0,1,2)).tolist(),
              'files':{p.name:sha(p) for p in sorted(dest.glob('*.npy'))}}
    write(dest/'identity.json',identity);return identity

class Data:
    def __init__(self,path,fold='main',asset=None):
        self.path=Path(path);self.identity=read(self.path/'identity.json');self.fold=fold
        self.dates=self.identity['dates'];self.assets=self.identity['assets'];self.asset=asset
        self.x=np.load(self.path/'x.npy',mmap_mode='r');self.y=np.load(self.path/'y.npy',mmap_mode='r')
        self.masks=np.load(self.path/'masks.npy',mmap_mode='r');self.cov=np.load(self.path/'covariates.npy',mmap_mode='r')
        self.bounds=SPLIT_INDICES[fold];a,tr,va,end=self.bounds
        self.ids={'train':list(range(a,tr)),'validation':list(range(tr,va)),
                  ('development_holdout' if fold=='main' else 'rolling_evaluation'):list(range(va,end))}
        self.scaler=TrainFactorScaler();raw=np.load(self.path/'raw_factors.npy',mmap_mode='r')
        self.scaler.fit(raw[a:tr]);self.factors=self.scaler.transform(raw)
        self.bins={}
        for c,name in enumerate(['relative_spread','l5_depth']):
            values=self.cov[a:tr,...,c];values=values[np.isfinite(values)]
            self.bins[name]=np.quantile(values,[1/3,2/3]).tolist()
        self.transform_hash=digest({'scaler':self.scaler.state(),'bins':self.bins,'train_dates':self.dates[a:tr]})
        self.split_hash=digest({'fold':fold,'bounds':self.bounds,'dates':self.dates})
        self.training_hash=digest({'train':self.dates[a:tr],'val':self.dates[tr:va],
                                   'input':self.identity['input_manifest_sha256'],'transform':self.transform_hash})
        self.asset_indices=list(range(27)) if asset is None else [self.assets.index(asset)]
    def batch(self,indices,device):
        ai=self.asset_indices
        def tensor(v,dtype=None):return torch.as_tensor(np.array(v[indices][:,:,ai],copy=True),dtype=dtype,device=device)
        return tensor(self.x),tensor(self.y,torch.long),tensor(self.masks[...,3],torch.bool),tensor(self.factors)
    def keys(self,split):
        ids=self.ids[split];assets=[self.assets[i] for i in self.asset_indices]
        index=pd.MultiIndex.from_product([[self.dates[i] for i in ids],GRID[:-1],assets],names=['date','origin_time','asset_id'])
        frame=index.to_frame(index=False);frame['endpoint_time']=np.repeat(np.tile(GRID[1:],len(ids)),len(assets))
        frame['split']=split;frame['fold']=self.fold
        frame['legacy_label']=self.y[ids][:,:,self.asset_indices].reshape(-1)
        for j,name in enumerate(POPS):frame[name]=self.masks[ids][:,:,self.asset_indices,j].reshape(-1).astype(np.int8)
        for j,name in enumerate(['relative_spread','l5_depth']):
            v=self.cov[ids][:,:,self.asset_indices,j].reshape(-1);cut=self.bins[name]
            frame[name+'_bin']=np.where(np.isfinite(v),np.where(v<=cut[0],'low',np.where(v<=cut[1],'middle','high')),'invalid')
        frame['origin_bucket']=frame.origin_time.str[:2]+':00–:59'
        return frame
