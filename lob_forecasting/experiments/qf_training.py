"""Matched real-data development trainer for the QF core decomposition."""
from __future__ import annotations
import csv,gzip,hashlib,json,os,random,subprocess,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from lob_forecasting.evaluation.qf_metrics import classification_metrics
from lob_forecasting.cli.parallel import bake_tensors
from lob_forecasting.data.preprocess import get_ticker_list
from lob_forecasting.experiments.shards import round_robin_shards,validate_shards
from lob_forecasting.models.qf_variants import build_qf_model

PRED_COLS=['date','timestamp','asset','chronological_fold','split','seed','model_id','protocol_id','horizon','representation','feature_set','true_class','predicted_class','probability_down','probability_flat','probability_up','validation_selected_checkpoint_epoch','original_current_price_flag','original_future_price_flag','stale_observation_flag','session_regime_flag']
TIMES=[f'{m//60:02d}:{m%60:02d}:00' for m in range(9*60,15*60+11,10)]

def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
def transformed_market_factors(x,eps=1e-9):
    """Target-excluding factors from contemporaneously available no-investor features."""
    logmid=x[...,0];ret=torch.zeros_like(logmid);ret[:,1:]=logmid[:,1:]-logmid[:,:-1]
    spread=torch.exp(x[...,1])-torch.exp(x[...,2])
    asks=torch.stack([torch.expm1(x[...,3+2*i]).clamp_min(0) for i in range(5)],-1)
    bids=torch.stack([torch.expm1(x[...,4+2*i]).clamp_min(0) for i in range(5)],-1)
    def obi(l):
        b=bids[...,:l].sum(-1);a=asks[...,:l].sum(-1);return (b-a)/(b+a+eps)
    vals=[ret,spread,obi(1),obi(5),(asks+bids).sum(-1),(ret>0).to(x.dtype)];n=x.shape[2]
    if n<2:raise ValueError('UM requires at least two assets')
    return torch.stack([(v.sum(2,keepdim=True)-v)/(n-1) for v in vals],-1)
def metrics(y,p):
    p=np.asarray(p,dtype=np.float64);p=p/p.sum(1,keepdims=True);result=classification_metrics(np.asarray(y),p);result['class_distribution']=np.bincount(np.asarray(y),minlength=3).tolist();return result
def forward(model,model_id,x):return model(x,transformed_market_factors(x)) if model_id=='UM' else model(x)
def evaluate(model,model_id,x,y,batch,device):
    model.eval();ps=[];ys=[]
    with torch.no_grad():
        for xb,yb in DataLoader(TensorDataset(x,y),batch_size=batch):
            logits=forward(model,model_id,xb.to(device));ps.append(logits.softmax(-1).cpu());ys.append(yb.cpu())
    return torch.cat(ys).numpy(),torch.cat(ps).numpy()
def fit(model_id,x,y,cfg,device,out,smoke=False):
    out.mkdir(parents=True,exist_ok=True);seed_all(int(cfg['seed']));model=build_qf_model(model_id,num_assets=x.shape[2],num_features=x.shape[3],d_model=cfg['d_model'],nhead=cfg['nhead'],temporal_layers=cfg['temporal_layers'],cross_asset_layers=cfg['cross_asset_layers'],dropout=cfg['dropout']).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);criterion=torch.nn.CrossEntropyLoss();n=len(x);tr=int(n*.7);va=int(n*.8);train=TensorDataset(x[:tr],y[:tr]);max_epochs=2 if smoke else cfg['max_epochs'];min_epochs=2 if smoke else cfg['min_epochs'];pat=2 if smoke else cfg['patience'];best_ll=float('inf');best_f1=-1.;bad=0;hist=[];start=1
    if (out/'last_checkpoint.pt').exists():
        resume=torch.load(out/'last_checkpoint.pt',map_location=device,weights_only=True);model.load_state_dict(resume['state_dict']);opt.load_state_dict(resume['optimizer']);start=resume['epoch']+1;best_ll=resume['best_ll'];best_f1=resume['best_f1'];bad=resume['bad'];hist=resume['history']
    for epoch in range(start,max_epochs+1):
        model.train();opt.zero_grad();losses=[]
        loader=DataLoader(train,batch_size=cfg['microbatch'],shuffle=True,generator=torch.Generator().manual_seed(cfg['seed']+epoch))
        for step,(xb,yb) in enumerate(loader,1):
            logits=forward(model,model_id,xb.to(device));loss=criterion(logits.reshape(-1,3),yb.to(device).reshape(-1))/cfg['accumulation'];loss.backward();losses.append(loss.item()*cfg['accumulation'])
            if step%cfg['accumulation']==0 or step==len(loader):torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);opt.step();opt.zero_grad()
        vy,vp=evaluate(model,model_id,x[tr:va],y[tr:va],cfg['microbatch'],device);m=metrics(vy.reshape(-1),vp.reshape(-1,3));hist.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_log_loss':m['log_loss'],'validation_macro_f1':m['macro_f1']})
        if m['log_loss']<best_ll-cfg['min_delta']:best_ll=m['log_loss'];bad=0;torch.save({'state_dict':model.state_dict(),'epoch':epoch},out/'best_logloss_checkpoint.pt')
        else:bad+=1
        if m['macro_f1']>best_f1:best_f1=m['macro_f1'];torch.save({'state_dict':model.state_dict(),'epoch':epoch},out/'best_macrof1_checkpoint.pt')
        torch.save({'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'best_ll':best_ll,'best_f1':best_f1,'bad':bad,'history':hist},out/'last_checkpoint.pt')
        if epoch>=min_epochs and bad>=pat:break
    ck=torch.load(out/'best_logloss_checkpoint.pt',map_location=device,weights_only=True);model.load_state_dict(ck['state_dict'],strict=True)
    pd.DataFrame(hist).to_csv(out/'training_history.csv',index=False);return model,ck['epoch'],(tr,va)
def prediction_rows(model,model_id,x,y,dates,tickers,split,epoch,cfg,device,start):
    yy,pp=evaluate(model,model_id,x,y,cfg['microbatch'],device);rows=[]
    for d in range(len(x)):
      for t in range(x.shape[1]):
       for a,ticker in enumerate(tickers):
        prob=pp[d,t,a];rows.append({'date':str(dates[start+d]),'timestamp':TIMES[t],'asset':ticker,'chronological_fold':'development','split':split,'seed':cfg['seed'],'model_id':model_id,'protocol_id':'qf_decomposition_dev_v1','horizon':1,'representation':'state','feature_set':'no_investor','true_class':int(yy[d,t,a]),'predicted_class':int(prob.argmax()),'probability_down':float(prob[0]),'probability_flat':float(prob[1]),'probability_up':float(prob[2]),'validation_selected_checkpoint_epoch':epoch,'original_current_price_flag':'unavailable','original_future_price_flag':'unavailable','stale_observation_flag':'unavailable','session_regime_flag':'unavailable'})
    return rows
def finalize(out,model_id,rows,cfg,dates,tickers):
    df=pd.DataFrame(rows,columns=PRED_COLS);df.to_csv(out/'test_predictions.csv.gz',index=False,compression='gzip');summ=[]
    for split,g in df.groupby('split'):summ.append({'scope':split,**metrics(g.true_class,g[['probability_down','probability_flat','probability_up']].to_numpy())})
    json.dump(summ,open(out/'metrics_summary.json','w'),indent=2);pd.DataFrame(summ).to_csv(out/'metrics_summary.csv',index=False)
    def grouped(cols,path):
      vals=[]
      for key,g in df.groupby(cols):vals.append({cols:key,**metrics(g.true_class,g[['probability_down','probability_flat','probability_up']].to_numpy())})
      pd.DataFrame(vals).to_csv(path,index=False)
    grouped('asset',out/'per_asset_metrics.csv');grouped('date',out/'per_date_metrics.csv');grouped('timestamp',out/'per_time_metrics.csv')
    conf=pd.crosstab(df.true_class,df.predicted_class).reindex(index=range(3),columns=range(3),fill_value=0);conf.to_csv(out/'confusion_matrix.csv');json.dump({'model':model_id,'state':'SUCCEEDED'},open(out/'status.json','w'));json.dump({'protocol':'qf_decomposition_dev_v1','feature_set':'no_investor','tickers':tickers},open(out/'manifest.json','w'));json.dump(cfg,open(out/'resolved_config.json','w'),indent=2);(out/'requested_config.yaml').write_text(Path('configs/qf/models/matched_temporal_v1.yaml').read_text());(out/'command.txt').write_text(' '.join(sys.argv)+'\n');(out/'environment.txt').write_text(sys.version+'\n');(out/'git_state.txt').write_text(subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout);json.dump({'dates':[str(x) for x in dates]},open(out/'data_manifest.json','w'));pd.DataFrame({'date':[str(x) for x in dates],'split':['train' if i<int(len(dates)*.7) else 'validation' if i<int(len(dates)*.8) else 'development_holdout' for i in range(len(dates))]}).to_csv(out/'split_manifest.csv',index=False);(out/'run.log').write_text('completed\n')
def run_shared(model_id,x,y,dates,tickers,cfg,device,out,smoke=False):
    model,epoch,(tr,va)=fit(model_id,x,y,cfg,device,out,smoke);rows=prediction_rows(model,model_id,x[tr:va],y[tr:va],dates,tickers,'validation',epoch,cfg,device,tr)+prediction_rows(model,model_id,x[va:],y[va:],dates,tickers,'development_holdout',epoch,cfg,device,va);finalize(out,model_id,rows,cfg,dates,tickers)
def run_s_shard(shard,x,y,dates,tickers,cfg,device,out,smoke=False):
    shards=round_robin_shards(tickers);validate_shards(shards);chosen=shards[shard];allrows=[]
    for ticker in chosen[:2] if smoke else chosen:
      i=tickers.index(ticker);sub=out/'assets'/ticker
      if (sub/'status.json').exists() and json.load(open(sub/'status.json')).get('state')=='SUCCEEDED':allrows.extend(pd.read_csv(sub/'test_predictions.csv.gz').to_dict('records'));continue
      model,epoch,(tr,va)=fit('S',x[:,:,i:i+1],y[:,:,i:i+1],cfg,device,sub,smoke);rows=prediction_rows(model,'S',x[tr:va,:,i:i+1],y[tr:va,:,i:i+1],dates,[ticker],'validation',epoch,cfg,device,tr)+prediction_rows(model,'S',x[va:,:,i:i+1],y[va:,:,i:i+1],dates,[ticker],'development_holdout',epoch,cfg,device,va);finalize(sub,'S',rows,cfg,dates,[ticker]);allrows+=rows
    pd.DataFrame(allrows,columns=PRED_COLS).to_csv(out/'test_predictions.csv.gz',index=False,compression='gzip');json.dump({'state':'SUCCEEDED','assets':chosen[:2] if smoke else chosen},open(out/'status.json','w'))
def load_data(data_dir,limit=None):
    tickers=get_ticker_list(data_dir);x,y,dates=bake_tensors(Path(data_dir),tickers,False,1e-4,torch.device('cpu'));return (x[:limit],y[:limit],dates[:limit],tickers) if limit else (x,y,dates,tickers)
