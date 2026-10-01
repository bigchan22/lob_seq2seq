"""Masked float32 trainer with exact valid-row accumulation and epoch-boundary resume."""
import inspect,os,random,time,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from common import BASE,PROBS,ROOT,read,write,sha,digest,code_sha,source_files,utc
from models import build,forward,ua_intervention
from data import Data
from scoring import save_predictions,validate_predictions,losses,analyze

def configure(seed):
    torch.set_num_threads(2)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
def load_checkpoint(path,device='cpu'):
    kw={'map_location':device}
    if 'weights_only' in inspect.signature(torch.load).parameters:kw['weights_only']=False
    return torch.load(path,**kw) # Only our task-local checkpoints, never arbitrary downloads.
def atomic_checkpoint(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(value,tmp);os.replace(tmp,path)
def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
def loss_sum(logits,y,mask):
    per=F.cross_entropy(logits.reshape(-1,3),y.reshape(-1),reduction='none')
    return (per*mask.reshape(-1)).sum(),int(mask.sum().item())
def step_window(model,opt,count,clip=1.):
    if count:
        for parameter in model.parameters():
            if parameter.grad is not None:parameter.grad.div_(count)
        torch.nn.utils.clip_grad_norm_(model.parameters(),clip);opt.step()
    opt.zero_grad(set_to_none=True)
def infer(model,name,data,indices,device,microbatch,condition=None):
    output=[];model.eval()
    with torch.no_grad():
        for i in range(0,len(indices),microbatch):
            x,y,mask,factors=data.batch(indices[i:i+microbatch],device)
            z=ua_intervention(model,x,condition) if condition else forward(model,name,x,factors)
            assert torch.isfinite(z).all(),'nonfinite logits'
            output.append(z.softmax(-1).cpu().numpy())
    return np.concatenate(output,0)
def prediction_frame(model,name,data,device,microbatch,seed,condition=None):
    frames=[]
    for split,ids in data.ids.items():
        if split=='train':continue
        frame=data.keys(split);p=infer(model,name,data,ids,device,microbatch,condition).reshape(-1,3)
        for i,col in enumerate(PROBS):frame[col]=p[:,i].astype(np.float64)
        frame['seed']=seed;frame['model']=name;frame['condition']=condition or 'original'
        frames.append(frame)
    return pd.concat(frames,ignore_index=True)

def train(config,cache,out,device,heartbeat=lambda **kwargs:None,resume=None,smoke=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);cfg=dict(BASE,**config)
    data=Data(cache,cfg.get('fold','main'),cfg.get('asset'));name=cfg['model'];seed=cfg['seed'];configure(seed)
    source=source_files(optional=name=='TLOB_ADAPTED')
    identity={'config':{k:v for k,v in cfg.items() if k not in ['source_job','select_from','family']},'protocol_hash':cfg['protocol_hash'],'smoke':smoke,
              'executable_source_digest':digest(source),'input_hash':data.identity['input_manifest_sha256'],
              'feature_hash':data.identity['files']['x.npy'],'split_hash':data.split_hash,'transform_hash':data.transform_hash}
    run_id=digest(identity);model=build(name,len(data.asset_indices)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    write(out/'resolved_config.json',dict(cfg,run_id=run_id,identity=identity,source_sha=code_sha(),source_files=source,
          parameters=parameters,training_hash=data.training_hash,scaler=data.scaler.state(),analysis_bins=data.bins,
          versions={'torch':torch.__version__,'numpy':np.__version__},precision='float32_tf32_disabled',
          source_dirty=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain']).strip()),
          accumulation_steps=32//cfg['microbatch'],smoke=smoke,
          observations_per_fit=int(data.masks[data.ids['train']][:,:,data.asset_indices,3].sum()),
          train_days=len(data.ids['train']),resume_semantics='exact saved epoch boundary; interrupted epoch recomputed from saved RNG/order'))
    best=float('inf');best_epoch=0;bad=0;history=[];start=1;best_state=None
    if resume and Path(resume).is_file():
        ck=load_checkpoint(resume)
        assert ck['run_id']==run_id,'incompatible resume identity'
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng'])
        best=ck['best'];best_epoch=ck['best_epoch'];best_state=ck['best_model'];bad=ck['bad'];history=ck['history'];start=ck['epoch']+1
    def checkpoint(epoch):
        atomic_checkpoint(out/'last_checkpoint.pt',dict(run_id=run_id,model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,
                          best=best,best_epoch=best_epoch,best_model=best_state,bad=bad,history=history,rng=rng_state(),
                          data_order={'next_epoch':epoch+1,'generator_seed':seed+epoch+1,'window_days':32},microbatch=cfg['microbatch']))
    checkpoint(start-1)
    max_epochs=2 if smoke else cfg['max_epochs'];train_ids=data.ids['train'][:4] if smoke else data.ids['train']
    val_ids=data.ids['validation'][:2] if smoke else data.ids['validation']
    last_checkpoint_time=time.time()
    for epoch in range(start,max_epochs+1):
        t0=time.monotonic();model.train();total_loss=0.;total_count=0;order=torch.randperm(len(train_ids),generator=torch.Generator().manual_seed(seed+epoch)).tolist()
        ordered=[train_ids[i] for i in order]
        for begin in range(0,len(ordered),32):
            window=ordered[begin:begin+32];opt.zero_grad(set_to_none=True);window_count=0
            for offset in range(0,len(window),cfg['microbatch']):
                x,y,mask,factors=data.batch(window[offset:offset+cfg['microbatch']],device)
                z=forward(model,name,x,factors);assert torch.isfinite(z).all(),'nonfinite training logits'
                value,count=loss_sum(z,y,mask);value.backward();window_count+=count
                total_loss+=float(value.detach());total_count+=count
            step_window(model,opt,window_count,cfg['gradient_clip_norm'])
            heartbeat(epoch=epoch,completed_day_panels=min(begin+32,len(ordered)),phase='train')
        probs=infer(model,name,data,val_ids,device,cfg['microbatch'])
        yy=np.array(data.y[val_ids][:,:,data.asset_indices]);mm=np.array(data.masks[val_ids][:,:,data.asset_indices,3])
        pp=probs[mm].astype(np.float64);pp/=pp.sum(1,keepdims=True)
        val_loss=float(-np.log(np.maximum(pp[np.arange(len(pp)),yy[mm]],np.finfo(np.float64).eps)).mean())
        assert np.isfinite(val_loss) and total_count>0,'invalid masked selection population'
        improved=val_loss<best-cfg['min_delta']
        if improved:
            best=val_loss;best_epoch=epoch;bad=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            atomic_checkpoint(out/'best_checkpoint.pt',dict(run_id=run_id,model=best_state,epoch=epoch,validation_p3_nll=best))
        else:bad+=1
        history.append(dict(epoch=epoch,train_p3_nll=total_loss/total_count,train_valid_rows=total_count,
                            validation_p3_nll=val_loss,validation_valid_rows=int(mm.sum()),accepted=improved,
                            selected_epoch=best_epoch,epoch_seconds=time.monotonic()-t0,microbatch=cfg['microbatch']))
        checkpoint(epoch);last_checkpoint_time=time.time()
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        heartbeat(epoch=epoch,phase='checkpoint',epoch_seconds=history[-1]['epoch_seconds'],checkpoint_time=last_checkpoint_time)
        print(__import__('json').dumps(dict(event='epoch',time=utc(),run_id=run_id,**history[-1])),flush=True)
        if epoch>=(1 if smoke else cfg['min_epochs']) and bad>=cfg['patience']:break
    assert best_state is not None
    model.load_state_dict(best_state)
    if not (out/'best_checkpoint.pt').exists():atomic_checkpoint(out/'best_checkpoint.pt',dict(run_id=run_id,model=best_state,epoch=best_epoch,validation_p3_nll=best))
    frame=prediction_frame(model,name,data,device,cfg['microbatch'],seed)
    expected=pd.concat([data.keys(s) for s in data.ids if s!='train'],ignore_index=True)
    validate_predictions(frame,expected)
    save_predictions(frame,out/'predictions.csv.gz')
    # No restricted score ever changes the already fixed checkpoint selection.
    summary=analyze(frame,out,{'model':name,'seed':seed,'family':cfg.get('family','new_p3')})
    result={'run_id':run_id,'source_sha':code_sha(),'protocol_hash':cfg['protocol_hash'],
            'executable_source_digest':identity['executable_source_digest'],'input_hash':identity['input_hash'],
            'split_hash':data.split_hash,'transform_hash':data.transform_hash,'training_hash':data.training_hash,
            'config_hash':digest(cfg),'model':name,'seed':seed,'lr':cfg['lr'],'fold':data.fold,'asset':cfg.get('asset'),
            'family':cfg.get('family'),'parameters':parameters,'selected_epoch':best_epoch,'validation_p3_nll':best,
            'epochs_completed':len(history),'prediction_rows':len(frame),'validated':True,
            'files':{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.suffix not in ['.pt','.tmp'] and p.name!='result.json'},
            'local_checkpoints':{'best_checkpoint.pt':sha(out/'best_checkpoint.pt'),'last_checkpoint.pt':sha(out/'last_checkpoint.pt')},
            'completed_at':utc(),'smoke':smoke}
    write(out/'result.json',result);return result
