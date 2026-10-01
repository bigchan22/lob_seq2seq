"""Foreground adapter for the already-published RTX3090 receiver interface.

Owns only local queue/processes/export files. The outer receiver is the sole Git
writer. All scientific work delegates to the common gated implementation.
"""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from queue_store import Queue,TERMINAL
from receiver_bridge.export import publication

GATES=['labels_masks_splits','causality','masked_loss_gradient','own_only_invariance','parameter_counts',
       'masked_accumulation_partial_window','train_only_scaling','r1_reuse_identity','restart_idempotency_stale_lease','real_data_smoke']

def verify(a):
    from data import prepare
    from gates import runtime_gate
    root=a.run_root;root.mkdir(parents=True,exist_ok=True);cache=root/'prepared'
    prepare(a.input_root,cache);runtime_gate(cache,ROOT)
    write(root/'receiver_verified.json',{'source_sha':code_sha(),'time':utc(),'cache':str(cache)})

def smoke(a):
    from gates import run,runtime_gate
    root=a.run_root;cache=root/'prepared';runtime_gate(cache,ROOT)
    # GPU0 is scoped to the receiver's explicitly allocated visible UUID list.
    device='cuda:0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
    result=run(cache,root/'gates',device)
    from scoring import validate_predictions
    from data import Data
    from jobs import read_predictions
    import pandas as pd
    d=Data(cache);expected=pd.concat([d.keys(s) for s in d.ids if s!='train'],ignore_index=True)
    validate_predictions(read_predictions(root/'gates/smoke_UA/predictions.csv.gz'),expected)
    ready=read(READY)
    write(root/'receiver_gate_report.json',{'protocol_sha256':ready['protocol_sha256'],'data_handoff_sha':HANDOFF,
          'gates':{name:True for name in GATES},'detailed_gate_path':'gates/gates.json','completed_at':utc()})

def run(a):
    root=a.run_root;root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    lock=(root/'bridge.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',');assert len(gpu)==a.workers and all(x.startswith('GPU-') for x in gpu)
    q=Queue(root)
    if not (root/'runtime.json').exists():
        write(root/'runtime.json',{'owner':a.owner,'source':str(ROOT),'source_sha':code_sha(),'cache':str(root/'prepared'),
              'gpu_uuids':gpu,'created_at':utc(),'publication':'outer_receiver_owned','python':sys.executable})
    q.initialize(read(CONFIG/'queue_manifest.json'),a.owner,ROOT)
    q.setmeta('stop',False);children={};handles=[];stopping=False
    def stop(signum,frame):
        nonlocal stopping
        stopping=True;q.setmeta('stop',True)
        for p in children.values():
            if p.poll() is None:p.terminate()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    specs=[('cpu','cpu','')]+[('gpu'+str(i),'gpu',g) for i,g in enumerate(gpu)]
    last=0;restarts={}
    while True:
        q.recover();q.refresh();status=q.snapshot();terminal=status['all_terminal']
        for name,res,g in specs:
            if not stopping and not terminal and not q.meta('stop',False) and (name not in children or children[name].poll() is not None):
                restarts[name]=restarts.get(name,-1)+1
                if restarts[name]>2:continue
                log=(root/'logs'/(name+'.log')).open('ab');handles.append(log)
                env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                # Same PGID as foreground bridge, so outer task-scoped stop covers
                # workers and their children. No unrelated process is signalled.
                children[name]=subprocess.Popen([sys.executable,'-u','-B',str(ROOT/'scripts/qf_overnight/runtime.py'),'worker',
                     '--root',str(root),'--resource',res,'--gpu',g],env=env,stdout=log,stderr=subprocess.STDOUT)
        status['workers']={n:{'pid':p.pid,'start':process_start(p.pid),'alive':p.poll() is None} for n,p in children.items()}
        write(root/'queue_status.json',status)
        if time.time()-last>60 or terminal:
            publication(root);last=time.time()
        if terminal or stopping or q.meta('stop',False):
            if all(p.poll() is not None for p in children.values()):publication(root);break
        time.sleep(10)
    q.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','smoke','run','pause','resume','retry'])
    p.add_argument('--run-root',type=Path,required=True);p.add_argument('--input-root');p.add_argument('--data-root')
    p.add_argument('--owner',default='rtx3090');p.add_argument('--workers',type=int,default=4);p.add_argument('--control-path');p.add_argument('--job')
    a=p.parse_args();assert a.owner=='rtx3090'
    if a.action=='verify':return verify(a)
    if a.action=='smoke':return smoke(a)
    if a.action=='run':return run(a)
    q=Queue(a.run_root)
    if a.action=='pause':q.setmeta('paused',True)
    if a.action=='resume':q.setmeta('paused',False);q.setmeta('stop',False)
    if a.action=='retry':
        r=q.get(a.job);assert r['state'] in ['BLOCKED','FAILED'] and r['retries']<2 and not alive(r['pid'],r['start'])
        q.db.execute("UPDATE jobs SET state='PENDING',retries=retries+1,not_before=0 WHERE id=?",(a.job,));q.event(a.job,'receiver_explicit_retry',{});q.db.commit()
    q.close()
if __name__=='__main__':main()
