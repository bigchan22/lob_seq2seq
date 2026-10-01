"""Finite DAG operators. Model selection reads validation P3 only."""
import argparse,json,os,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from common import *
from queue_store import Queue
from data import Data
from scoring import analyze,save_predictions,validate_predictions,metrics,paired_bootstrap

def artifact(q,id):
    row=q.get(id);assert row['state']=='SUCCEEDED',id
    result=read(row['result'])
    return artifact(q,result['reused_from']) if result.get('reused_from') else (Path(row['result']).parent,result)
def chosen(q,selector):return q.result(selector)['selected_job']
def read_predictions(path):return pd.read_csv(path,dtype={'asset_id':str,'date':str,'origin_time':str,'endpoint_time':str})
def selection(q,job,out):
    trials=[]
    for id in job['config']['trials']:
        directory,result=artifact(q,id)
        trials.append({k:result[k] for k in ['model','seed','lr','validation_p3_nll','selected_epoch','run_id','source_sha','protocol_hash']}|{'job_id':id})
    best=min(x['validation_p3_nll'] for x in trials)
    winner=min((x for x in trials if x['validation_p3_nll']<=best+1e-8),key=lambda x:x['lr'])
    result={'validated':True,'selected_job':winner['job_id'],'lr':winner['lr'],'model':winner['model'],
            'criterion':'selected-checkpoint validation P3 NLL, tie<=1e-8 smaller LR','trials':trials}
    write(out/'selection.json',result);return result
def prior(cache,out,fold,protocol):
    data=Data(cache,fold);ids=data.ids['train'];y=np.array(data.y[ids]);mask=np.array(data.masks[ids,...,3])
    counts=np.bincount(y[mask],minlength=3);global_p=(counts+1)/(counts.sum()+3)
    per=np.stack([np.bincount(y[:,:,i][mask[:,:,i]],minlength=3) for i in range(27)])
    per_p=(per+1)/(per.sum(1,keepdims=True)+3)
    for name in ['global','per_asset']:
        frames=[]
        for split in data.ids:
            if split=='train':continue
            frame=data.keys(split)
            p=np.tile(global_p,(len(frame),1)) if name=='global' else np.tile(per_p,(len(frame)//27,1))
            for j,col in enumerate(PROBS):frame[col]=p[:,j]
            frame['seed']=0;frame['model']='prior_'+name
            frames.append(frame)
        frame=pd.concat(frames,ignore_index=True);save_predictions(frame,out/('predictions_'+name+'.csv.gz'))
        analyze(frame,out/name,dict(model='prior_'+name,seed=0,family='train_P3_Laplace_prior'))
    return {'validated':True,'fold':fold,'protocol_hash':protocol,'training_hash':data.training_hash,
            'global_train_counts':counts.tolist(),'per_asset_train_counts':per.tolist(),'laplace_alpha':1,'fit_population':'P3 train only'}
def historical(cache,out):
    index=read(ROOT/'docs/qf_data_review/20261001/prediction_index.json');data=Data(cache)
    expected=pd.concat([data.keys(s) for s in ['validation','development_holdout']],ignore_index=True)
    references={('UA','validation'):.887593050,('UC','validation'):.903287486,
                ('UA','development_holdout'):.908952596,('UC','development_holdout'):.922755800}
    checked=[]
    for record in index['selected']:
        path=ROOT/record['prediction_path'];assert sha(path)==record['prediction_sha256']
        p=pd.read_csv(path,dtype={'asset':str,'date':str,'timestamp':str})
        p=p.rename(columns={'asset':'asset_id','timestamp':'origin_time'});p['date']=pd.to_datetime(p.date,format='%Y%m%d').dt.strftime('%Y-%m-%d')
        p['fold']='main';assert not p.duplicated(KEYS).any()
        frame=expected.merge(p[KEYS+PROBS+['true_class']],on=KEYS,how='outer',validate='one_to_one',indicator=True)
        assert frame['_merge'].eq('both').all() and frame.legacy_label.eq(frame.true_class).all()
        frame['seed']=record['seed'];frame['model']=record['model']
        if record['seed']==42 and record['model'] in ['UA','UC']:
            for split,g in frame.groupby('split'):
                nll=metrics(g)['nll'];ref=references[(record['model'],split)];assert abs(nll-ref)<1e-7,(record['model'],split,nll,ref)
                checked.append(dict(model=record['model'],split=split,nll=nll,reference=ref,error=nll-ref))
        analyze(frame,out/(record['model']+'_'+str(record['seed'])),dict(model=record['model'],seed=record['seed'],family='historical_august_selected'))
    return {'validated':True,'historical_only':True,'frozen_index_sha256':sha(ROOT/'docs/qf_data_review/20261001/prediction_index.json'),'anchors':checked,'files':9}

def execute(q,row,out,device):
    job=json.loads(row['payload']);cfg=job['config'];runtime=read(q.root/'runtime.json');cache=runtime['cache'];kind=job['kind']
    if kind=='gate':
        from gates import runtime_gate
        return runtime_gate(cache,ROOT)
    if kind=='prior':return prior(cache,out,cfg['fold'],job['protocol_hash'])
    if kind=='historical':return historical(cache,out)
    if kind=='select':return selection(q,job,out)
    if kind=='alias':
        source=cfg.get('source_job') or chosen(q,cfg['select_from']);directory,result=artifact(q,source)
        assert result['validated'];assert sha(directory/'predictions.csv.gz')==result['files']['predictions.csv.gz']
        return dict(result,reused_from=source,reuse_reason='exact existing fit, no new optimization or inference')
    if kind in ['fit','fixed_or_alias']:
        from trainer import train
        config=dict(cfg,protocol_hash=job['protocol_hash'])
        if cfg.get('select_from'):config['lr']=q.result(cfg['select_from'])['lr']
        if kind=='fixed_or_alias':
            directory,result=artifact(q,cfg['source_job'])
            if result['lr']==cfg['lr']:return dict(result,reused_from=cfg['source_job'],reuse_reason='fixed LR equals already completed selected LR')
        if row['oom']:config['microbatch']=4
        resume=None
        if row['attempt']>1 and not row['oom']:
            earlier=sorted((q.root/'runs'/job['job_id']).glob('attempt_*/last_checkpoint.pt'))
            if earlier:resume=earlier[-1]
        result=train(config,cache,out,device,heartbeat=lambda **kw:q.heartbeat(job['job_id'],True,**kw),resume=resume)
        result['resume_checkpoint_used']=str(resume.relative_to(q.root)) if resume else None
        result['attempt']=row['attempt'];return result
    if kind=='analysis':
        directory,result=artifact(q,cfg['source_job']);frame=read_predictions(directory/'predictions.csv.gz')
        data=Data(cache,result['fold'],result.get('asset'))
        expected=pd.concat([data.keys(s) for s in data.ids if s!='train'],ignore_index=True);validate_predictions(frame,expected)
        analyze(frame,out,dict(model=result['model'],seed=result['seed'],family='new_P3_selected',source_job=cfg['source_job']))
        return {'validated':True,'source_job':cfg['source_job'],'prediction_sha256':sha(directory/'predictions.csv.gz')}
    if kind=='rolling_reuse':
        directory,result=artifact(q,cfg['source_job']);data=Data(cache,'R1')
        assert result['training_hash']==data.training_hash,'R1 train/val/transform reuse mismatch'
        frame=read_predictions(directory/'predictions.csv.gz');allowed=[data.dates[i] for v in data.ids.values() for i in v]
        frame=frame[frame.date.isin(allowed)].copy();frame['fold']='R1';frame['split']=frame.split.replace({'development_holdout':'rolling_evaluation'})
        expected=pd.concat([data.keys(s) for s in data.ids if s!='train'],ignore_index=True);validate_predictions(frame,expected)
        save_predictions(frame,out/'predictions.csv.gz');analyze(frame,out,dict(model=cfg['model'],seed=cfg['seed'],family='retrospective_rolling'))
        return dict(result,reused_main_job=cfg['source_job'],fold='R1',prediction_rows=len(frame),validated=True,
                    files={'predictions.csv.gz':sha(out/'predictions.csv.gz')},reuse_reason='same train/val, LR, seed, configuration, scaler; eval subset only')
    if kind=='pool':
        parts=[];parameters=0;run_ids=[]
        for id in cfg['source_jobs']:
            directory,result=artifact(q,id);parts.append(read_predictions(directory/'predictions.csv.gz'));parameters+=result['parameters'];run_ids.append(result['run_id'])
        frame=pd.concat(parts,ignore_index=True);data=Data(cache)
        expected=pd.concat([data.keys(s) for s in data.ids if s!='train'],ignore_index=True);validate_predictions(frame,expected)
        save_predictions(frame,out/'predictions.csv.gz');analyze(frame,out,dict(model='S',seed=cfg['seed'],family='fixed_decomposition'))
        comparisons=[]
        u0=read_predictions(artifact(q,cfg['fixed_jobs']['U0'])[0]/'predictions.csv.gz')
        u1=read_predictions(artifact(q,cfg['fixed_jobs']['U1'])[0]/'predictions.csv.gz')
        from scoring import losses
        for name,left,right in [('fixed_U0_minus_S',u0,frame),('fixed_U1_minus_U0',u1,u0)]:
            for split in ['validation','development_holdout']:
                for pop in POPS:
                    a=left[left.split.eq(split)&left[pop].eq(1)].sort_values(KEYS).reset_index(drop=True)
                    b=right[right.split.eq(split)&right[pop].eq(1)].sort_values(KEYS).reset_index(drop=True)
                    assert a[KEYS+['legacy_label']].equals(b[KEYS+['legacy_label']])
                    comparisons.append(dict(comparison=name,seed=cfg['seed'],split=split,population=pop,rows=len(a),delta_nll=float((losses(a)[0]-losses(b)[0]).mean())))
        pd.DataFrame(comparisons).to_csv(out/'paired_differences.csv',index=False)
        return {'validated':True,'model':'S','seed':cfg['seed'],'parameters_total':parameters,'separate_parameter_sets':27,
                'run_ids':run_ids,'family':'fixed_decomposition','unequal_individual_vs_pooled_fit_budgets':True,'prediction_rows':len(frame)}
    if kind=='intervention':
        from trainer import configure,load_checkpoint,prediction_frame
        from models import build
        import torch
        directory,result=artifact(q,cfg['source_job']);data=Data(cache);configure(cfg['seed'])
        model=build('UA').to(device);checkpoint=load_checkpoint(directory/'best_checkpoint.pt');model.load_state_dict(checkpoint['model'])
        original=read_predictions(directory/'predictions.csv.gz');support=original.P3_intersection.eq(1)&original.origin_time.gt('09:00:00')
        original=original[support].copy().sort_values(KEYS).reset_index(drop=True)
        from scoring import losses
        differences=[]
        for condition in ['self_only','peer_lag1']:
            frame=prediction_frame(model,'UA',data,device,4,cfg['seed'],condition)
            frame=frame[frame.P3_intersection.eq(1)&frame.origin_time.gt('09:00:00')].sort_values(KEYS).reset_index(drop=True)
            assert original[KEYS+['legacy_label']].equals(frame[KEYS+['legacy_label']])
            save_predictions(frame,out/(condition+'.csv.gz'));analyze(frame,out/condition,dict(model='UA',seed=cfg['seed'],family='distribution_shift_sensitivity',condition=condition))
            delta=losses(frame)[0]-losses(original)[0]
            for split,g in frame.assign(delta=delta).groupby('split'):
                differences.append(dict(seed=cfg['seed'],condition=condition,split=split,rows=len(g),delta_nll=float(g.delta.mean()),support='common P3, original full temporal history'))
        pd.DataFrame(differences).to_csv(out/'paired_differences.csv',index=False)
        return {'validated':True,'source_job':cfg['source_job'],'diagnostic':'inference-only distribution shift; not trained lag model or causal attribution'}
    if kind=='final':
        snapshot=q.snapshot();write(out/'local_final_snapshot.json',snapshot)
        return {'validated':True,'local_queue_terminal':True,'counts_at_finalization':snapshot['counts'],'peer_final_not_required':True}
    raise ValueError('Unimplemented finite job kind: '+kind)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--job',required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    q=Queue(a.root);row=q.get(a.job);q.bind(a.job,os.getpid());out=a.root/'runs'/a.job/('attempt_%04d'%row['attempt']);out.mkdir(parents=True,exist_ok=True)
    try:
        result=execute(q,row,out,a.device);result.update(job_id=a.job,owner=q.meta('owner'),protocol_hash=q.meta('protocol_hash'))
        result.setdefault('source_sha',code_sha());result.setdefault('completed_at',utc())
        write(out/'result.json',result);q.success(a.job,out/'result.json')
    except Exception as error:
        traceback.print_exc();message=str(error)
        category='oom' if 'out of memory' in message.lower() else 'transient' if isinstance(error,(TimeoutError,ConnectionError)) else 'scientific'
        write(out/'failure.json',{'category':category,'type':type(error).__name__,'message':message,'source_sha':code_sha(),'time':utc()})
        q.fail(a.job,message,category);raise
    finally:q.close()

if __name__=='__main__':main()
