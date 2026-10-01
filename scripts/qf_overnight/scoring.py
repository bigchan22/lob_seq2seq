"""Auditable row-weighted metrics, calibration, heterogeneity and date-panel bootstrap."""
import gzip,io
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef
from common import PROBS,POPS,KEYS,write

def normalized(frame):
    p=frame[PROBS].to_numpy(np.float64)
    assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
    sums=p.sum(1);assert (sums>0).all()
    return p/sums[:,None],float(np.max(np.abs(sums-1)))
def losses(frame):
    p,err=normalized(frame);y=frame.legacy_label.to_numpy(int)
    assert set(y)<={0,1,2}
    return -np.log(np.maximum(p[np.arange(len(y)),y],np.finfo(np.float64).eps)),p,err
def metrics(frame):
    ll,p,err=losses(frame);y=frame.legacy_label.to_numpy(int);pred=p.argmax(1)
    return dict(rows=len(y),nll=float(ll.mean()),accuracy=float(accuracy_score(y,pred)),
                macro_f1=float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),
                mcc=float(matthews_corrcoef(y,pred)),brier=float(((p-np.eye(3)[y])**2).sum(1).mean()),
                equal_asset_nll=float(pd.DataFrame({'asset':frame.asset_id.to_numpy(),'loss':ll}).groupby('asset').loss.mean().mean()),
                down=int((y==0).sum()),flat=int((y==1).sum()),up=int((y==2).sum()),max_probability_sum_error=err)
def save_predictions(frame,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    assert not frame.duplicated(KEYS).any()
    with path.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as z:
            with io.TextIOWrapper(z,encoding='utf-8',newline='') as text:frame.to_csv(text,index=False,lineterminator='\n',float_format='%.17g')
def validate_predictions(frame,expected):
    assert len(frame)==len(expected) and not frame.duplicated(KEYS).any()
    a=frame.sort_values(KEYS).reset_index(drop=True);b=expected.sort_values(KEYS).reset_index(drop=True)
    assert a[KEYS+['endpoint_time','legacy_label']].equals(b[KEYS+['endpoint_time','legacy_label']])
    for pop in POPS:assert np.array_equal(a[pop].to_numpy(),b[pop].to_numpy())
    normalized(a);return True
def analyze(frame,out,identity):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    summary=[];perasset=[];subgroups=[];cal=[];daily=[]
    for (fold,split),group in frame.groupby(['fold','split'],sort=True):
        for pop in POPS:
            g=group[group[pop]==1].copy()
            if not len(g):continue
            base=dict(identity,fold=fold,split=split,population=pop)
            summary.append(dict(base,**metrics(g)))
            ll,p,_=losses(g);g['loss']=ll
            for asset,part in g.groupby('asset_id'):perasset.append(dict(base,asset_id=asset,**metrics(part)))
            for col in ['origin_bucket','relative_spread_bin','l5_depth_bin']:
                for value,part in g.groupby(col):subgroups.append(dict(base,grouping=col,group=value,**metrics(part)))
            for date,part in g.groupby('date'):daily.append(dict(base,date=date,loss_sum=float(part.loss.sum()),rows=len(part)))
            confidence=p.max(1);correct=p.argmax(1)==g.legacy_label.to_numpy(int)
            bins=np.minimum((confidence*15).astype(int),14);ece=0
            for i in range(15):
                keep=bins==i;n=int(keep.sum());acc=float(correct[keep].mean()) if n else None;conf=float(confidence[keep].mean()) if n else None
                if n:ece+=n/len(g)*abs(acc-conf)
                cal.append(dict(base,bin=i,lower=i/15,upper=(i+1)/15,rows=n,accuracy=acc,mean_confidence=conf))
            summary[-1]['ece_15']=ece
    for name,records in [('metrics',summary),('per_asset',perasset),('subgroup_metrics',subgroups),('calibration',cal),('daily_losses',daily)]:
        pd.DataFrame(records).to_csv(out/(name+'.csv'),index=False)
    # Dependency-free SVG reliability plot, descriptive only; no fitted calibrator.
    pts=[]
    for r in cal:
        if r['population']=='P3_intersection' and r['rows'] and r['split']!='validation':
            pts.append('<circle cx="%.2f" cy="%.2f" r="4" fill="#2166ac"><title>%s n=%d</title></circle>'%(30+300*r['mean_confidence'],330-300*r['accuracy'],r['split'],r['rows']))
    svg='<svg xmlns="http://www.w3.org/2000/svg" width="380" height="380"><rect width="100%" height="100%" fill="white"/><path d="M30 30V330H330 M30 330L330 30" stroke="gray" fill="none"/>'+''.join(pts)+'<text x="40" y="360">P3 reliability: confidence vs accuracy</text></svg>'
    (out/'reliability.svg').write_text(svg)
    return summary

def paired_bootstrap(left,right,draws=10000,block=5,seed=20261001):
    # Multiple seeds stay paired inside dates. Summing per-seed losses gives the
    # mean of seed losses after division by the identically repeated row count.
    key=KEYS+['seed'];a=left.sort_values(key).reset_index(drop=True);b=right.sort_values(key).reset_index(drop=True)
    assert not a.duplicated(key).any() and a[key+['legacy_label']].equals(b[key+['legacy_label']])
    la,_,_=losses(a);lb,_,_=losses(b);d=a[['fold','date','seed']].copy();d['delta']=la-lb
    by=d.groupby(['fold','date']).delta.agg(['sum','size']);rng=np.random.default_rng(seed)
    totals=np.zeros(draws);counts=np.zeros(draws)
    for fold,g in by.groupby(level=0,sort=True):
        sums=g['sum'].to_numpy();sizes=g['size'].to_numpy();n=len(g)
        # Non-circular moving blocks; truncate the concatenated sequence to n.
        length=min(block,n);starts=rng.integers(0,n-length+1,size=(draws,(n+length-1)//length))
        ix=(starts[:,:,None]+np.arange(length)).reshape(draws,-1)[:,:n]
        totals+=sums[ix].sum(1);counts+=sizes[ix].sum(1)
    values=totals/counts
    return {'rows_including_seeds':len(a),'delta':float((la-lb).mean()),'ci_low':float(np.quantile(values,.025)),
            'ci_high':float(np.quantile(values,.975)),'draws':draws,'block_dates':block,'seed':seed,
            'per_seed':{str(s):float(g.delta.mean()) for s,g in d.groupby('seed')},'status':'exploratory_retrospective'}
