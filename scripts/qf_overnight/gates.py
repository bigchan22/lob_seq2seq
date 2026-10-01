"""Mandatory bounded scientific and queue gates; no performance/improvement gate."""
import argparse,copy,json,os,tempfile,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from common import *
from data import Data
from models import build,forward,cross_layer
from trainer import loss_sum,step_window,configure,train
from queue_store import Queue

def accumulation_check():
    configure(17);reference=nn.Linear(13,3).double();tested=copy.deepcopy(reference)
    x=torch.randn(37,2,3,13,dtype=torch.float64);y=torch.randint(0,3,(37,2,3));mask=torch.rand(37,2,3)>.3
    mask[0]=False;mask[-1]=False
    a=torch.optim.SGD(reference.parameters(),lr=.01);b=torch.optim.SGD(tested.parameters(),lr=.01)
    for start,end in [(0,32),(32,37)]:
        a.zero_grad();value,count=loss_sum(reference(x[start:end]),y[start:end],mask[start:end]);(value/count).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(),1.);a.step()
        b.zero_grad();total=0
        for begin in range(start,end,8):
            stop=min(begin+8,end);value,count=loss_sum(tested(x[begin:stop]),y[begin:stop],mask[begin:stop]);value.backward();total+=count
        step_window(tested,b,total)
        for p,q in zip(reference.parameters(),tested.parameters()):assert torch.allclose(p,q,rtol=1e-11,atol=1e-12)
    logits=torch.randn(2,4,3,requires_grad=True);labels=torch.randint(0,3,(2,4));m=torch.tensor([[True,False,True,False],[False,True,False,True]])
    loss_sum(logits,labels,m)[0].backward();assert logits.grad[~m].eq(0).all()
    return 'PASS: uneven masked batches, all-excluded micro rows and final5-day partial32-day window; zero excluded-output gradient'

def queue_check():
    with tempfile.TemporaryDirectory(prefix='qf_queue_gate_') as d:
        q=Queue(d);jobs=[dict(job_id='one',owner='a5000',kind='gate',resource='cpu',priority=0,deps=[],config={}),
                         dict(job_id='two',owner='a5000',kind='gate',resource='cpu',priority=0,deps=['one'],config={})]
        manifest={'protocol_hash':'fixture','jobs':jobs};q.initialize(manifest,'a5000',ROOT);q.initialize(manifest,'a5000',ROOT)
        assert len(q.rows())==2;one=q.claim('cpu');assert one['id']=='one' and q.claim('cpu') is None
        q.db.execute("UPDATE jobs SET heartbeat=0 WHERE id='one'");q.db.commit();q.recover();assert q.get('one')['state']=='RUNNING'
        q.db.execute("UPDATE jobs SET pid=0,start='',heartbeat=0 WHERE id='one'");q.db.commit();q.recover();assert q.get('one')['state']=='PENDING'
        q.db.execute("UPDATE jobs SET not_before=0 WHERE id='one'");q.db.commit();q.claim('cpu')
        bad=Path(d)/'bad.json';write(bad,{'validated':False})
        try:q.success('one',bad)
        except AssertionError:pass
        else:raise AssertionError('unvalidated success accepted')
        good=Path(d)/'good.json';write(good,{'validated':True});q.success('one',good)
        assert q.claim('cpu')['id']=='two';q.close()
    return 'PASS: idempotent init, atomic dependency claim, live-process stale protection, dead-process retry, validation required'

def runtime_gate(cache,source):
    protocol=read(DOCS/'protocol.json');ready=read(READY)
    assert digest(protocol)==ready['protocol_hash'] and ready['READY_CORE']
    assert source_files()==ready['core_source_files'],'execution source differs from frozen READY'
    identity=read(Path(cache)/'identity.json')
    assert identity['input_manifest_sha256']==protocol['input_manifest_sha256']
    assert identity['forecast_rows_sha256']==protocol['forecast_rows_sha256'] and identity['label_mismatches']==0
    assert identity['mask_counts']==[505818,469391,439263,432560]
    for name,value in identity['files'].items():assert sha(Path(cache)/name)==value
    # Feature/factor bytes are scientific inputs. A receiver with incompatible
    # arithmetic must report the mismatch rather than silently mixing datasets.
    assert identity['files']==ready['prepared_data_hashes'],'prepared feature/factor bytes differ from frozen common implementation'
    return {'validated':True,'protocol_hash':digest(protocol),'core_digest':ready['core_executable_digest'],
            'label_and_mask_checks':'PASS','prepared_data_hashes_match':True}

def run(cache,out,device='cpu',optional=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);configure(42);data=Data(cache)
    x,y,mask,factors=data.batch([20,21],device)
    results={'accumulation':accumulation_check(),'queue':queue_check(),'parameters':{},'causality':{},'smokes':{}}
    models=['U1','UC','UA','UM_V2_DEPTH','UX','GRU_U1','SHARED_QUERY','U0','S'] if not optional else ['TLOB_ADAPTED']
    for name in models:
        configure(42);xx=x[:,:,:1] if name=='S' else x;model=build(name,xx.shape[2]).to(device).eval()
        with torch.no_grad():
            baseline=forward(model,name,xx,factors)
            changed=xx.clone();changed[:,20:]+=.31
            changed_factors=factors.clone();changed_factors[:,20:]+=.31
            altered=forward(model,name,changed,changed_factors)
            assert baseline.shape==(*xx.shape[:3],3) and torch.isfinite(baseline).all()
            error=float((baseline[:,:20]-altered[:,:20]).abs().max());assert error<3e-5,(name,'future',error)
            changed=xx.clone();changed[1]+=.2
            no_cross=forward(model,name,changed,factors)
            assert torch.allclose(baseline[0],no_cross[0],atol=3e-5,rtol=1e-5),(name,'day boundary')
            if name in ['U0','U1','UC','GRU_U1']:
                changed=xx.clone();changed[:,:,1:]+=.5
                own=forward(model,name,changed,factors)
                assert torch.allclose(baseline[:,:,0],own[:,:,0],atol=2e-6,rtol=1e-6),(name,'peer access')
        count=sum(p.numel() for p in model.parameters() if p.requires_grad);results['parameters'][name]=count
        results['causality'][name]={'max_past_logit_change_after_future_perturbation':error,'day_isolation':True}
        del model
    if not optional:
        assert results['parameters']['UA']==results['parameters']['UC']==results['parameters']['SHARED_QUERY']==605315
        ua=build('UA').to(device).eval();sq=build('SHARED_QUERY').to(device).eval();sq.load_state_dict(ua.state_dict())
        with torch.no_grad():
            h=ua.encode(x);b,t,n,d=h.shape;flat=h.reshape(b*t,n,d);layer=ua.cross_asset_encoder.layers[0]
            manual=ua.output_head(cross_layer(layer,flat,flat,flat,flat).reshape(b,t,n,d))
            assert torch.allclose(manual,ua(x),atol=3e-6,rtol=2e-6),'manual cross path not equivalent to original UA'
        from lob_forecasting.experiments.discrimination_factors import leave_one_out,TrainFactorScaler
        a=x.detach().cpu();b=a.clone();b[:,:,0]+=.1
        original=leave_one_out(a);altered=leave_one_out(b)
        assert torch.allclose(original[:,:,0],altered[:,:,0],atol=2e-5,rtol=2e-5),'UM target exclusion'
        raw=np.load(Path(cache)/'raw_factors.npy');mutated=raw.copy();mutated[345:]*=10
        first=TrainFactorScaler().fit(raw[:345]).state();second=TrainFactorScaler().fit(mutated[:345]).state();assert first==second
        assert Data(cache,'main').training_hash==Data(cache,'R1').training_hash
        results['UM_target_exclusion']='PASS within float32 summation tolerance'
        results['train_only_scaling_and_R1_reuse']='PASS'
    protocol_hash=digest(read(DOCS/'protocol.json'))
    for name in models:
        conf=dict(model=name,seed=42,lr=.0001,fold='main',family='bounded_gate_smoke',protocol_hash=protocol_hash)
        if name=='S':conf['asset']=data.assets[0]
        record=train(conf,cache,out/('smoke_'+name),device,smoke=True)
        assert record['validated'] and record['epochs_completed']==2
        results['smokes'][name]={k:record[k] for k in ['parameters','selected_epoch','epochs_completed','prediction_rows','validated']}
    results.update(passed=True,training_smokes_only=True,device=device,torch=torch.__version__,numpy=np.__version__,
                   protocol_hash=protocol_hash,source_files=source_files(optional),completed_at=utc())
    write(out/'gates.json',results);print(json.dumps({'passed':True,'parameters':results['parameters'],'out':str(out)}));return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cpu');p.add_argument('--optional',action='store_true');a=p.parse_args();run(a.cache,a.out,a.device,a.optional)
