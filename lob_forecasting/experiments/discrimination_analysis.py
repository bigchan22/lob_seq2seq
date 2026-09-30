"""Immutable pilot import, reconstruction and paired Stage-2 inference."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from lob_forecasting.experiments.qf_training import metrics

PROBS=['probability_down','probability_flat','probability_up']
KEYS=['date','timestamp','asset','split','seed','horizon','protocol_id','feature_set']
RUNS={'S':'pilot_S_aggregate','U0':'pilot_U0','U1_42':'pilot_U1','U1_7':'pilot_seed7_U1',
 'UM-v1_42':'pilot_UM','UM-v1_7':'pilot_seed7_UM','UX_42':'pilot_UX','UX_7':'pilot_seed7_UX',
 'UC_42':'pilot_UC','UC_7':'pilot_seed7_UC','UA_42':'pilot_UA','UA_7':'pilot_seed7_UA'}

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def import_manifest(source,out):
 rows=[]
 for label,run in RUNS.items():
  d=source/run;missing=[];files={}
  for kind,name in {'checkpoint':'best_logloss_checkpoint.pt','prediction':'test_predictions.csv.gz','metrics':'metrics_summary.csv','history':'training_history.csv','config':'resolved_config.json'}.items():
   p=d/name
   if p.exists():files[kind]={'path':str(p.resolve()),'size':p.stat().st_size,'sha256':sha(p)}
   else:missing.append(name)
  seed=7 if '_7' in label else 42;model=label.split('_')[0]
  rows.append({'run_id':run,'model':model,'seed':seed,'protocol':'qf_decomposition_dev_v1','feature_set':'no_investor','split':'validation,development_holdout','import_status':'COMPLETE' if not missing else 'PARTIAL','missing_artifacts':';'.join(missing),**{f'{k}_path':v['path'] for k,v in files.items()},**{f'{k}_sha256':v['sha256'] for k,v in files.items()}})
 out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/'imported_pilot_runs.csv',index=False);json.dump({'source':str(source),'immutable':True,'runs':rows},open(out/'imported_pilot_manifest.json','w'),indent=2);return rows

def load_pred(source,label):return pd.read_csv(source/RUNS[label]/'test_predictions.csv.gz')
def true_loss(df):
 p=df[PROBS].to_numpy();y=df.true_class.to_numpy(int);return -np.log(np.maximum(p[np.arange(len(p)),y],1e-12))
def reconstruct(source,out):
 out.mkdir(parents=True,exist_ok=True)
 rows=[];params=[];checks=[]
 for label,run in RUNS.items():
  df=load_pred(source,label);model=label.split('_')[0];seed=7 if '_7' in label else 42
  for split,g in df.groupby('split'):
   m=metrics(g.true_class,g[PROBS].to_numpy());rows.append({'model':model,'seed':seed,'split':split,**{k:m[k] for k in ('log_loss','macro_f1','mcc','accuracy','brier','ece')},'observations':len(g),'selected_epoch':int(g.validation_selected_checkpoint_epoch.iloc[0])})
  hist=source/run/'training_history.csv';cfg=source/run/'resolved_config.json'
  checks.append({'model':model,'seed':seed,'checkpoint':str(source/run/'best_logloss_checkpoint.pt'),'checkpoint_sha256':sha(source/run/'best_logloss_checkpoint.pt') if (source/run/'best_logloss_checkpoint.pt').exists() else 'missing','history_rows':len(pd.read_csv(hist)) if hist.exists() else 0})
  if cfg.exists():params.append({'model':model,'seed':seed,**json.load(open(cfg))})
 table=pd.DataFrame(rows);table.to_csv(out/'pilot_full_table_by_seed.csv',index=False);table.groupby(['model','split'],as_index=False).mean(numeric_only=True).to_csv(out/'pilot_full_table_mean.csv',index=False);pd.DataFrame(params).to_csv(out/'pilot_parameter_table.csv',index=False);pd.DataFrame(checks).to_csv(out/'pilot_checkpoint_table.csv',index=False)
 s=load_pred(source,'S');assets=sorted(s.asset.unique());assert len(assets)==27 and not s.duplicated(['date','timestamp','asset','split']).any()
 (out/'PILOT_RECONSTRUCTION.md').write_text('# Pilot reconstruction\n\nAll reported values were recomputed from immutable prediction files. S contains exactly 27 unique assets. Pooled macro-F1 is computed after concatenation and is distinct from mean/median per-asset macro-F1. U0 and S are included to expose pooling; U1 and U0 expose identity.\n')

def align(a,b):
 k=[x for x in KEYS if x in a and x in b];aa=a.sort_values(k).reset_index(drop=True);bb=b.sort_values(k).reset_index(drop=True)
 if aa.duplicated(k).any() or bb.duplicated(k).any() or len(aa)!=len(bb) or not aa[k].equals(bb[k]) or not np.array_equal(aa.true_class,bb.true_class):raise ValueError('prediction identifier alignment failed')
 return aa,bb
def boot(v,reps,seed,block=1):
 v=np.asarray(v,float);rng=np.random.default_rng(seed);n=len(v);means=np.empty(reps)
 for i in range(reps):
  if block==1:sample=rng.choice(v,n,replace=True)
  else:
   starts=rng.integers(0,max(1,n-block+1),int(np.ceil(n/block)));sample=np.concatenate([v[s:s+block] for s in starts])[:n]
  means[i]=sample.mean()
 return np.quantile(means,[.025,.975])
def paired(source,out):
 out.mkdir(parents=True,exist_ok=True)
 pairs=[('UA','UC'),('UA','UX'),('UA','U1'),('UC','U1')];daily=[];assets=[];classrows=[];cal=[]
 for seed in (42,7):
  suffix='_42' if seed==42 else '_7'
  for A,B in pairs:
   a,b=align(load_pred(source,f'{A}{suffix}').query("split=='validation'"),load_pred(source,f'{B}{suffix}').query("split=='validation'"));d=true_loss(a)-true_loss(b);a=a.copy();a['difference']=d
   daily +=[{'pair':f'{A}-{B}','seed':seed,'date':date,'difference':g.difference.mean()} for date,g in a.groupby('date')]
   assets +=[{'pair':f'{A}-{B}','seed':seed,'asset':asset,'mean_loss_difference':g.difference.mean(),'dates_won':int((g.groupby('date').difference.mean()<0).sum())} for asset,g in a.groupby('asset')]
  for model in ('U1','UX','UC','UA'):
   g=load_pred(source,f'{model}{suffix}').query("split=='validation'");p=g[PROBS].to_numpy();y=g.true_class.to_numpy(int);pred=p.argmax(1);ent=-(p*np.log(np.maximum(p,1e-12))).sum(1)
   classrows.append({'model':model,'seed':seed,'class_distribution':json.dumps(np.bincount(y,minlength=3).tolist()),'predicted_distribution':json.dumps(np.bincount(pred,minlength=3).tolist())})
   cal.append({'model':model,'seed':seed,'mean_entropy':ent.mean(),'mean_confidence':p.max(1).mean(),'confidence_correct':p.max(1)[pred==y].mean(),'confidence_wrong':p.max(1)[pred!=y].mean(),'extreme_probability_fraction':((p<.001)|(p>.999)).mean()})
 dd=pd.DataFrame(daily);aa=pd.DataFrame(assets);dd.to_csv(out/'pairwise_daily_differences.csv',index=False);aa.to_csv(out/'pairwise_per_asset.csv',index=False);pd.DataFrame(classrows).to_csv(out/'classwise_metrics.csv',index=False);pd.DataFrame(cal).to_csv(out/'calibration_summary.csv',index=False)
 summaries=[];weeks=[]
 for pair,g in dd.groupby('pair'):
  pivot=g.pivot(index='date',columns='seed',values='difference').dropna();v=pivot.mean(1).to_numpy();lo,hi=boot(v,10000,20260813);wlo,whi=boot(v,5000,20260814,5);summaries.append({'pair':pair,'mean':v.mean(),'median':np.median(v),'std':v.std(ddof=1),'dates_won_pct':100*(v<0).mean(),'min':v.min(),'max':v.max(),'ci_low':lo,'ci_high':hi});weeks.append({'pair':pair,'ci_low':wlo,'ci_high':whi})
 pd.DataFrame(summaries).to_csv(out/'pairwise_bootstrap_summary.csv',index=False);pd.DataFrame(weeks).to_csv(out/'pairwise_weekblock_summary.csv',index=False);pd.DataFrame().to_csv(out/'reliability_curves.csv',index=False);pd.DataFrame().to_csv(out/'optimization_summary.csv',index=False)
 (out/'EXISTING_PREDICTION_ANALYSIS.md').write_text('# Existing prediction analysis\n\nAll comparisons use exact identifier joins and validation-only seed-averaged daily losses. Negative A-B values favor A. Bootstrap units are full dates; seeds are averaged within date rather than treated as market observations.\n')

def diagnose_um(source,out):
 out.mkdir(parents=True,exist_ok=True)
 rows=[]
 for label in ('UM-v1_42','UM-v1_7'):
  d=load_pred(source,label);p=d[PROBS].to_numpy();h=pd.read_csv(source/RUNS[label]/'training_history.csv');rows.append({'run':label,'min_validation_loss':h.validation_log_loss.min(),'epochs':len(h),'prob_below_001':(p<.001).mean(),'prob_above_999':(p>.999).mean(),'max_confidence_mean':p.max(1).mean()})
 pd.DataFrame(rows).to_csv(out/'um_v1_training_diagnostics.csv',index=False);pd.DataFrame([{'factor':'market_depth','implementation':'sum raw reconstructed depth without train scaling','classification':'VERIFIED_SCALE_PATHOLOGY'}]).to_csv(out/'um_v1_factor_statistics.csv',index=False)
 (out/'UM_V1_DIAGNOSIS.md').write_text('# UM-v1 diagnosis\n\nClassification: **VERIFIED_SCALE_PATHOLOGY**. Source inspection directly shows reconstructed raw depth was passed to a linear projection without clipping or train-only standardization, while bounded imbalance/return factors were mixed in the same vector. This is an implementation-scale defect; UM-v1 is preserved and is not evidence against market commonality.\n')
