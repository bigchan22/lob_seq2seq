"""Executable real-data QF development jobs and CPU aggregations."""
import argparse,json,os,shutil
from pathlib import Path
import pandas as pd
import torch,yaml
from .queue_db import QueueDB
from .qf_training import load_data,run_shared,run_s_shard,metrics
from .pilot_gate import decide

def config(seed=42):
 c=yaml.safe_load(Path('configs/qf/models/matched_temporal_v1.yaml').read_text());return {'seed':int(seed),'d_model':c['d_model'],'nhead':c['nhead'],'temporal_layers':c['temporal_layers'],'cross_asset_layers':c['cross_asset_layers'],'dropout':c['dropout'],'learning_rate':c['learning_rate'],'weight_decay':c['weight_decay'],'max_epochs':c['max_epochs'],'min_epochs':c['min_epochs'],'patience':c['early_stopping_patience'],'min_delta':c['early_stopping_min_delta'],'microbatch':c['microbatch_size'],'accumulation':c['gradient_accumulation_steps'],'gradient_clip':c['gradient_clip_norm']}
def aggregate_s(root,out):
 files=list(root.glob('pilot_S_shard_*/assets/*/test_predictions.csv.gz'));frames=[pd.read_csv(x) for x in files];df=pd.concat(frames,ignore_index=True);assets=sorted(df.asset.unique());
 if len(assets)!=27:raise RuntimeError(f'S aggregation requires 27 assets, found {len(assets)}')
 out.mkdir(parents=True,exist_ok=True);df.to_csv(out/'test_predictions.csv.gz',index=False,compression='gzip');rows=[]
 for asset,g in df[df.split=='validation'].groupby('asset'):rows.append({'asset':asset,**metrics(g.true_class,g[['probability_down','probability_flat','probability_up']].to_numpy())})
 pd.DataFrame(rows).to_csv(out/'per_asset_metrics.csv',index=False);json.dump({'state':'SUCCEEDED','assets':assets},open(out/'status.json','w'))
def compare(root,out):
 ids=['pilot_S_aggregate','pilot_U0','pilot_U1','pilot_UM','pilot_UX','pilot_UC','pilot_UA'];rows=[];asset=[]
 for jid in ids:
  df=pd.read_csv(root/jid/'test_predictions.csv.gz');v=df[df.split=='validation'];m=metrics(v.true_class,v[['probability_down','probability_flat','probability_up']].to_numpy());rows.append({'model':jid.replace('pilot_',''),**m})
  for a,g in v.groupby('asset'):asset.append({'model':jid.replace('pilot_',''),'asset':a,**metrics(g.true_class,g[['probability_down','probability_flat','probability_up']].to_numpy())})
 out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/'pilot_model_summary.csv',index=False);pd.DataFrame(asset).to_csv(out/'pilot_per_asset_differences.csv',index=False);pd.DataFrame(rows).to_csv(out/'pilot_pairwise_comparisons.csv',index=False);json.dump({'state':'SUCCEEDED'},open(out/'status.json','w'))
def gate(root,out):
 s=pd.read_csv(root/'pilot_compare/pilot_model_summary.csv').set_index('model');a=pd.read_csv(root/'pilot_compare/pilot_per_asset_differences.csv');ua=a[a.model=='UA'].set_index('asset');um=a[a.model=='UM'].set_index('asset');wins=int((ua.log_loss<um.log_loss).sum());status=decide(s.loc['UA'],s.loc['UM'],s.loc['UX'],s.loc['UC'],wins,True);out.mkdir(parents=True,exist_ok=True);json.dump({'status':status,'ua_vs_um_asset_wins':wins},open(out/'pilot_gate_decision.json','w'),indent=2);(out/'PILOT_REPORT.md').write_text(f'# Pilot gate\n\nStatus: **{status}**; UA wins: {wins}/27.\n');json.dump({'state':'SUCCEEDED'},open(out/'status.json','w'));return status
def compare_seed7(root,out):
 models=['U1','UM','UX','UC','UA'];summary=[];per_asset=[]
 for seed,prefix in [(42,'pilot_'),(7,'pilot_seed7_')]:
  for model in models:
   df=pd.read_csv(root/f'{prefix}{model}'/'test_predictions.csv.gz');v=df[df.split=='validation'];summary.append({'seed':seed,'model':model,**metrics(v.true_class,v[['probability_down','probability_flat','probability_up']].to_numpy())})
   for asset,g in v.groupby('asset'):per_asset.append({'seed':seed,'model':model,'asset':asset,**metrics(g.true_class,g[['probability_down','probability_flat','probability_up']].to_numpy())})
 s=pd.DataFrame(summary);a=pd.DataFrame(per_asset);numeric=['log_loss','macro_f1','mcc','accuracy','brier','ece','prediction_count'];avg=s.groupby('model',as_index=False)[numeric].mean();aa=a.groupby(['model','asset'],as_index=False)[numeric].mean();ua=aa[aa.model=='UA'].set_index('asset');um=aa[aa.model=='UM'].set_index('asset');wins=int((ua.log_loss<um.log_loss).sum());idx=avg.set_index('model');status=decide(idx.loc['UA'],idx.loc['UM'],idx.loc['UX'],idx.loc['UC'],wins,True)
 out.mkdir(parents=True,exist_ok=True);s.to_csv(out/'seed_metrics.csv',index=False);avg.to_csv(out/'seed_averaged_model_summary.csv',index=False);aa.to_csv(out/'seed_averaged_per_asset_metrics.csv',index=False);json.dump({'status':status,'ua_vs_um_asset_wins':wins,'decision_basis':'mean validation metrics across seeds 42 and 7'},open(out/'pilot_gate_decision.json','w'),indent=2);json.dump({'state':'SUCCEEDED'},open(out/'status.json','w'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--job-id',required=True);a=p.parse_args();art=Path(os.environ.get('QF_ARTIFACT_ROOT','artifacts'));root=art/'runs';out=root/a.job_id
 if a.job_id.startswith('preflight_'):return 0
 if a.job_id=='pilot_S_aggregate':aggregate_s(root,out);return 0
 if a.job_id=='pilot_compare':compare(root,out);return 0
 if a.job_id=='pilot_gate':gate(root,out);return 0
 if a.job_id=='pilot_seed7_compare':compare_seed7(root,out);return 0
 if a.job_id=='pilot_package':out.mkdir(parents=True,exist_ok=True);json.dump({'state':'SUCCEEDED'},open(out/'status.json','w'));return 0
 db=QueueDB(art/'queue/qf_queue.sqlite');db.initialize();row=next((r for r in db.rows() if r['job_id']==a.job_id),None)
 if not row:raise SystemExit('job not found')
 if row['protocol']!='qf_decomposition_dev_v1' or row['feature_set']!='no_investor':raise SystemExit('runner supports only development no-investor protocol')
 smoke=row['phase']=='smoke';device=torch.device(os.environ.get('QF_LOCAL_DEVICE','cuda:0') if torch.cuda.is_available() else 'cpu');x,y,dates,tickers=load_data(Path(os.environ.get('QF_DATA_DIR','data/processed')),12 if smoke else None);cfg=config(row['seed'])
 model=row['model'];
 if model=='S':run_s_shard(row['asset_shard'] or 'A',x,y,dates,tickers,cfg,device,out,smoke)
 else:run_shared(model,x,y,dates,tickers,cfg,device,out,smoke)
if __name__=='__main__':main()
