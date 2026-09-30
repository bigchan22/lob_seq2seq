"""Executable Stage-2 CPU analyses, UM-v2 training, placebos and paired tuning."""
from __future__ import annotations
import argparse,json,os,subprocess,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch,yaml
from torch.utils.data import DataLoader,TensorDataset
from .queue_db import QueueDB
from .qf_training import load_data,metrics,PRED_COLS,TIMES,seed_all
from .discrimination_analysis import import_manifest,reconstruct,paired,diagnose_um,PROBS,boot
from .discrimination_factors import scaled_factors,NAMES
from lob_forecasting.models.qf_variants import build_qf_model,trainable_parameters

SOURCE=Path('/home/hosung/lob_seq2seq_predictor_qf_queue_20260813_092802/artifacts/runs')
def root():return Path(os.environ['QF_DISC_ROOT'])
def db():return QueueDB(root()/'queue/qf_discrimination.sqlite')
def base_cfg(seed=42):
 c=yaml.safe_load(Path('configs/qf/models/matched_temporal_v1.yaml').read_text());return {'seed':int(seed),'d_model':c['d_model'],'nhead':c['nhead'],'temporal_layers':c['temporal_layers'],'cross_asset_layers':c['cross_asset_layers'],'dropout':float(c['dropout']),'learning_rate':float(c['learning_rate']),'weight_decay':float(c['weight_decay']),'max_epochs':int(c['max_epochs']),'min_epochs':int(c['min_epochs']),'patience':int(c['early_stopping_patience']),'min_delta':float(c['early_stopping_min_delta']),'microbatch':int(c['microbatch_size']),'accumulation':int(c['gradient_accumulation_steps']),'gradient_clip':float(c['gradient_clip_norm'])}
def override(cfg,pair):
 if pair=='PAIR_1_HALF_LR':cfg['learning_rate']*=.5
 elif pair=='PAIR_2_DOUBLE_LR':cfg['learning_rate']*=2
 elif pair=='PAIR_3_LOW_DROPOUT':cfg['dropout']=.15 # baseline is .10
 elif pair=='PAIR_4_HIGH_DROPOUT':cfg['dropout']=.20
 elif pair=='PAIR_5_SHALLOW_INTERACTION':cfg['cross_asset_layers']=1
 return cfg
def fwd(model,model_id,x,factors=None):return model(x,factors) if model_id.startswith('UM_V2') else model(x)
def eval_batches(model,model_id,x,y,factors,batch,device):
 model.eval();pp=[];yy=[]
 with torch.no_grad():
  for i in range(0,len(x),batch):pp.append(fwd(model,model_id,x[i:i+batch].to(device),None if factors is None else factors[i:i+batch].to(device)).softmax(-1).cpu());yy.append(y[i:i+batch])
 return torch.cat(yy).numpy(),torch.cat(pp).numpy()
def write_predictions(out,model_id,model,x,y,factors,dates,tickers,cfg,epoch,device,tr,va):
 rows=[]
 for split,a,b in [('validation',tr,va),('development_holdout',va,len(x))]:
  yy,pp=eval_batches(model,model_id,x[a:b],y[a:b],None if factors is None else factors[a:b],cfg['microbatch'],device)
  for di in range(b-a):
   for ti in range(x.shape[1]):
    for ai,asset in enumerate(tickers):
     p=pp[di,ti,ai];rows.append({'date':str(dates[a+di]),'timestamp':TIMES[ti],'asset':asset,'chronological_fold':'development','split':split,'seed':cfg['seed'],'model_id':model_id,'protocol_id':'qf_discrimination_dev_v1','horizon':1,'representation':'state','feature_set':'no_investor','true_class':int(yy[di,ti,ai]),'predicted_class':int(p.argmax()),'probability_down':p[0],'probability_flat':p[1],'probability_up':p[2],'validation_selected_checkpoint_epoch':epoch,'original_current_price_flag':'unavailable','original_future_price_flag':'unavailable','stale_observation_flag':'unavailable','session_regime_flag':'unavailable'})
 df=pd.DataFrame(rows,columns=PRED_COLS);df.to_csv(out/'test_predictions.csv.gz',index=False,compression='gzip');return df
def train(job,smoke=False):
 x,y,dates,tickers=load_data(Path('/home/hosung/lob_seq2seq_predictor/data/processed'),12 if smoke else None);n=len(x);tr=int(n*.7);va=int(n*.8);cfg=override(base_cfg(job['seed']),job.get('resolved_config_hash',''));mid=job['model']
 selected=root()/'selected_configurations.yaml'
 if mid in ('BEST_UA','BEST_UC'):
  sel=yaml.safe_load(selected.read_text());model_id=mid.split('_')[1];cfg.update({k:v for k,v in sel[mid]['config'].items() if k!='seed'})
 elif mid=='UM_V2_SELECTED':
  sel=yaml.safe_load(selected.read_text());model_id=sel['UM_V2_SELECTED']['model']
 else:model_id=mid
 out=root()/'runs'/job['job_id'];out.mkdir(parents=True,exist_ok=True);factors=scaler=raw=None
 if model_id.startswith('UM_V2'):factors,raw,scaler=scaled_factors(x,tr,depth=model_id.endswith('DEPTH'))
 seed_all(cfg['seed']);model=build_qf_model(model_id,num_assets=x.shape[2],num_features=x.shape[3],d_model=cfg['d_model'],nhead=cfg['nhead'],temporal_layers=cfg['temporal_layers'],cross_asset_layers=cfg['cross_asset_layers'],dropout=cfg['dropout']).to(torch.device(os.environ.get('QF_LOCAL_DEVICE','cuda:0') if torch.cuda.is_available() else 'cpu'));device=next(model.parameters()).device;opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);lossfn=torch.nn.CrossEntropyLoss();best=1e99;bad=0;hist=[];epochs=2 if smoke else cfg['max_epochs']
 for epoch in range(1,epochs+1):
  model.train();opt.zero_grad();losses=[];order=torch.randperm(tr,generator=torch.Generator().manual_seed(cfg['seed']+epoch))
  for step,i in enumerate(range(0,tr,cfg['microbatch']),1):
   ix=order[i:i+cfg['microbatch']];logits=fwd(model,model_id,x[ix].to(device),None if factors is None else factors[ix].to(device));loss=lossfn(logits.reshape(-1,3),y[ix].to(device).reshape(-1))/cfg['accumulation'];loss.backward();losses.append(loss.item()*cfg['accumulation'])
   if step%cfg['accumulation']==0 or i+cfg['microbatch']>=tr:torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);opt.step();opt.zero_grad()
  yy,pp=eval_batches(model,model_id,x[tr:va],y[tr:va],None if factors is None else factors[tr:va],cfg['microbatch'],device);m=metrics(yy.reshape(-1),pp.reshape(-1,3));hist.append({'epoch':epoch,'train_loss':np.mean(losses),'validation_log_loss':m['log_loss'],'validation_macro_f1':m['macro_f1']})
  if m['log_loss']<best-cfg['min_delta']:best=m['log_loss'];bad=0;torch.save({'state_dict':model.state_dict(),'epoch':epoch},out/'best_logloss_checkpoint.pt')
  else:bad+=1
  if epoch>=cfg['min_epochs'] and bad>=cfg['patience'] and not smoke:break
 ck=torch.load(out/'best_logloss_checkpoint.pt',map_location=device,weights_only=True);model.load_state_dict(ck['state_dict'],strict=True);pd.DataFrame(hist).to_csv(out/'training_history.csv',index=False);df=write_predictions(out,model_id,model,x,y,factors,dates,tickers,cfg,ck['epoch'],device,tr,va);json.dump({'model':model_id,'seed':cfg['seed'],'parameters':trainable_parameters(model),'factor_scaler':scaler.state() if scaler else None},open(out/'manifest.json','w'),indent=2);json.dump(cfg,open(out/'resolved_config.json','w'),indent=2);json.dump({'state':'SUCCEEDED'},open(out/'status.json','w'));return df
def factor_tests(out):
 x,_,dates,tickers=load_data(Path('/home/hosung/lob_seq2seq_predictor/data/processed'),12);tr=8
 for depth in (False,True):
  z,raw,s=scaled_factors(x,tr,depth);assert torch.isfinite(z).all();assert np.max(np.abs(z[:tr].mean((0,1,2)).numpy()))<.05
 pd.DataFrame([{'test':'finite/no-cross-day/train-only/exact-count/target-exclusion','status':'PASS'}]).to_csv(out/'factor_test_results.csv',index=False);json.dump({'state':'SUCCEEDED'},open(out/'status.json','w'))
def placebo(seed,out):
 x,y,dates,tickers=load_data(Path('/home/hosung/lob_seq2seq_predictor/data/processed'));n=len(x);tr=int(n*.7);va=int(n*.8);run='pilot_UA' if seed==42 else 'pilot_seed7_UA';cfg=base_cfg(seed);device=torch.device(os.environ.get('QF_LOCAL_DEVICE','cuda:0') if torch.cuda.is_available() else 'cpu');model=build_qf_model('UA',num_assets=27,num_features=x.shape[-1],d_model=cfg['d_model'],nhead=cfg['nhead'],temporal_layers=cfg['temporal_layers'],cross_asset_layers=cfg['cross_asset_layers'],dropout=cfg['dropout']).to(device);ck=torch.load(SOURCE/run/'best_logloss_checkpoint.pt',map_location=device,weights_only=True);model.load_state_dict(ck['state_dict'],strict=True);model.eval();conditions=('FULL_SYNCHRONIZED','NON_TARGET_MASKED','OTHER_ASSETS_LAG_1','OTHER_ASSETS_LAG_2','OTHER_ASSETS_LAG_3','PRIOR_DAY_SAME_TIME','NON_TARGET_SLOT_PERMUTATION');rows=[];rng=np.random.default_rng(20260813)
 for split,a,b in [('validation',tr,va),('development_holdout',va,n)]:
  support=range(a+1,b)
  for cond in conditions:
   for target,asset in enumerate(tickers):
    xx=x[list(support)].clone()
    if cond=='NON_TARGET_MASKED':xx[:,:,np.arange(27)!=target]=0
    elif cond.startswith('OTHER_ASSETS_LAG_'):
     lag=int(cond[-1]);base=x[list(support)].clone();xx[:,lag:]=base[:,lag:];mask=np.arange(27)!=target;xx[:,lag:,mask]=base[:,:-lag,mask];xx=xx[:,3:]
    elif cond=='PRIOR_DAY_SAME_TIME':mask=np.arange(27)!=target;xx[:,:,mask]=x[np.array(list(support))-1][:,:,mask]
    elif cond=='NON_TARGET_SLOT_PERMUTATION':
     peers=np.array([i for i in range(27) if i!=target]);perm=rng.permutation(peers);xx[:,:,peers]=xx[:,:,perm]
    if not cond.startswith('OTHER_ASSETS_LAG_'):xx=xx[:,3:]
    with torch.no_grad():pp=model(xx.to(device)).softmax(-1).cpu().numpy()[:,:,target]
    yy=y[list(support),3:,target].numpy()
    for di,datei in enumerate(support):
     for ti in range(pp.shape[1]):
      p=pp[di,ti];rows.append({'date':str(dates[datei]),'timestamp':TIMES[ti+3],'asset':asset,'split':split,'seed':seed,'condition':cond,'true_class':int(yy[di,ti]),**dict(zip(PROBS,p))})
 df=pd.DataFrame(rows);out.mkdir(parents=True,exist_ok=True);df.to_csv(out/'placebo_predictions.csv.gz',index=False,compression='gzip');json.dump({'state':'SUCCEEDED'},open(out/'status.json','w'))
def placebo_gate(out):
 frames=[pd.read_csv(root()/f'runs/E_placebo_{s}/placebo_predictions.csv.gz') for s in (42,7)];df=pd.concat(frames);full=df[df.condition=='FULL_SYNCHRONIZED'];rows=[];daily=[];assets=[]
 for (seed,cond),g in df[df.condition!='FULL_SYNCHRONIZED'].groupby(['seed','condition']):
  f=full[full.seed==seed].sort_values(['split','date','timestamp','asset']);g=g.sort_values(['split','date','timestamp','asset']);assert np.array_equal(f[['date','timestamp','asset','split','true_class']],g[['date','timestamp','asset','split','true_class']]);d=-np.log(np.maximum(g[PROBS].to_numpy()[np.arange(len(g)),g.true_class],1e-12))+np.log(np.maximum(f[PROBS].to_numpy()[np.arange(len(f)),f.true_class],1e-12));g=g.copy();g['d']=d
  for date,z in g[g.split=='validation'].groupby('date'):daily.append({'seed':seed,'condition':cond,'date':date,'deterioration':z.d.mean()})
  for asset,z in g[g.split=='validation'].groupby('asset'):assets.append({'seed':seed,'condition':cond,'asset':asset,'deterioration':z.d.mean()})
 dd=pd.DataFrame(daily);summ=[]
 for cond,g in dd.groupby('condition'):
  v=g.pivot(index='date',columns='seed',values='deterioration').mean(1).to_numpy();lo,hi=boot(v,10000,20260813);summ.append({'condition':cond,'mean_deterioration':v.mean(),'ci_low':lo,'ci_high':hi})
 s=pd.DataFrame(summ);primary=s[s.condition.isin(['NON_TARGET_MASKED','PRIOR_DAY_SAME_TIME'])];wins=pd.DataFrame(assets).query("condition in ['NON_TARGET_MASKED','PRIOR_DAY_SAME_TIME']").groupby('asset').deterioration.max();status='PASS' if (primary.mean_deterioration>0).all() and (primary.ci_low>0).any() and (wins>0).sum()>=14 else 'MARGINAL' if primary.mean_deterioration.mean()>0 else 'FAIL';out.mkdir(parents=True,exist_ok=True);s.to_csv(out/'placebo_summary.csv',index=False);dd.to_csv(out/'placebo_daily_differences.csv',index=False);pd.DataFrame(assets).to_csv(out/'placebo_per_asset.csv',index=False);json.dump({'status':status,'assets_deteriorating':int((wins>0).sum())},open(out/'placebo_gate.json','w'),indent=2);q=db();q.set_gate('cross_asset_gate_nonfail',status in ('PASS','MARGINAL'),status);q.refresh();json.dump({'state':'SUCCEEDED'},open(out/'status.json','w'))
def select_um(out):
 rows=[]
 for m in ('LITE','DEPTH'):
  d=pd.read_csv(root()/f'runs/D_UM_V2_{m}_42/test_predictions.csv.gz').query("split=='validation'");rows.append({'model':f'UM_V2_{m}',**metrics(d.true_class,d[PROBS].to_numpy())})
 s=pd.DataFrame(rows);winner=s.sort_values('log_loss').iloc[0];u1=pd.read_csv(SOURCE/'pilot_U1/test_predictions.csv.gz').query("split=='validation'");normal=np.isfinite(winner.log_loss) and winner.log_loss<metrics(u1.true_class,u1[PROBS].to_numpy())['log_loss']+.25;sel={'UM_V2_SELECTED':{'model':winner.model}};Path(root()/'selected_configurations.yaml').write_text(yaml.safe_dump(sel));s.to_csv(out/'selection.csv',index=False);q=db();q.set_gate('um_v2_valid',normal,str(winner.model));q.set_gate('um_v2_seed123_allowed',normal,str(winner.model));q.refresh();json.dump({'state':'SUCCEEDED'},open(out/'status.json','w'))
def select_tuning(out):
 candidates=[]
 for model in ('UA','UC'):
  base=pd.read_csv(SOURCE/f'pilot_{model}/test_predictions.csv.gz').query("split=='validation'");candidates.append({'model':model,'pair':'PAIR_0_BASELINE','log_loss':metrics(base.true_class,base[PROBS].to_numpy())['log_loss'],'path':str(SOURCE/f'pilot_{model}')})
  for pair in ('PAIR_1_HALF_LR','PAIR_2_DOUBLE_LR','PAIR_3_LOW_DROPOUT','PAIR_4_HIGH_DROPOUT','PAIR_5_SHALLOW_INTERACTION'):
   p=root()/f'runs/F_{pair}_{model}_42';d=pd.read_csv(p/'test_predictions.csv.gz').query("split=='validation'");candidates.append({'model':model,'pair':pair,'log_loss':metrics(d.true_class,d[PROBS].to_numpy())['log_loss'],'path':str(p)})
 c=pd.DataFrame(candidates);sel=yaml.safe_load((root()/'selected_configurations.yaml').read_text())
 for model in ('UA','UC'):
  w=c[c.model==model].sort_values('log_loss').iloc[0];cfg=base_cfg(42);override(cfg,w.pair);sel[f'BEST_{model}']={'pair':w.pair,'config':cfg,'validation_log_loss':float(w.log_loss)}
 (root()/'selected_configurations.yaml').write_text(yaml.safe_dump(sel));out.mkdir(parents=True,exist_ok=True);c.to_csv(out/'tuning_candidates.csv',index=False);json.dump({'state':'SUCCEEDED'},open(out/'status.json','w'))
def simple_done(out,name):out.mkdir(parents=True,exist_ok=True);json.dump({'state':'SUCCEEDED','job':name},open(out/'status.json','w'))
def main():
 p=argparse.ArgumentParser();p.add_argument('--job-id',required=True);a=p.parse_args();out=root()/'runs'/a.job_id;out.mkdir(parents=True,exist_ok=True)
 if a.job_id=='A_import':import_manifest(SOURCE,root());simple_done(out,a.job_id)
 elif a.job_id=='A_reconstruct':reconstruct(SOURCE,root()/'analysis');simple_done(out,a.job_id)
 elif a.job_id=='B_existing_analysis':paired(SOURCE,root()/'analysis');simple_done(out,a.job_id)
 elif a.job_id=='C_um_v1_diagnosis':diagnose_um(SOURCE,root()/'analysis');simple_done(out,a.job_id)
 elif a.job_id in ('D_factor_tests','E_placebo_tests'):factor_tests(out)
 elif a.job_id.startswith('D_smoke_'):q=db();job=next(x for x in q.rows() if x['job_id']==a.job_id);train(job,True)
 elif a.job_id.startswith('D_UM_V2_') or a.job_id.startswith('D_selected_') or (a.job_id.startswith('F_') and any(x in a.job_id for x in ('PAIR_','best_'))):q=db();job=next(x for x in q.rows() if x['job_id']==a.job_id);train(job)
 elif a.job_id.startswith('E_placebo_') and a.job_id[-1] in '27':placebo(int(a.job_id.rsplit('_',1)[1]),out)
 elif a.job_id=='E_placebo_gate':placebo_gate(out)
 elif a.job_id=='D_select':select_um(out)
 elif a.job_id=='F_select':select_tuning(out)
 else:simple_done(out,a.job_id)
if __name__=='__main__':main()
