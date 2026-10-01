"""Task-scoped detached supervisor, workers and controls. No global queue assumptions."""
import argparse,fcntl,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
from common import ROOT,CONFIG,READY,read,write,utc,code_sha,process_start,alive
from queue_store import Queue,TERMINAL

def worker(root,resource,gpu):
    root=Path(root);lock=(root/'locks');lock.mkdir(exist_ok=True)
    handle=(lock/(resource+'_'+str(gpu).replace('/','_')+'.lock')).open('a')
    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    os.environ['CUDA_VISIBLE_DEVICES']=gpu if resource=='gpu' else ''
    q=Queue(root)
    while not q.meta('stop',False):
        free=shutil.disk_usage(root).free
        if free<10*(1<<30):q.setmeta('disk_pressure',{'free_bytes':free,'claims_paused':True,'time':utc()});time.sleep(30);continue
        if q.meta('disk_pressure'):q.setmeta('disk_pressure',None)
        row=q.claim(resource)
        if not row:
            if all(r['state'] in TERMINAL for r in q.rows()):break
            time.sleep(5);continue
        out=root/'runs'/row['id']/('attempt_%04d'%row['attempt']);out.mkdir(parents=True,exist_ok=True)
        with (out/'process.log').open('ab',buffering=0) as log:
            command=[sys.executable,'-u','-B',str(Path(row['source'])/'scripts/qf_overnight/jobs.py'),
                     '--root',str(root),'--job',row['id'],'--device','cuda:0' if resource=='gpu' else 'cpu']
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
            q.bind(row['id'],child.pid)
            while child.poll() is None:
                q.heartbeat(row['id']);current=q.get(row['id']);detail=json.loads(current['detail'] or '{}')
                epoch_seconds=float(detail.get('epoch_seconds',0));limit=max(3600.,20*epoch_seconds) if epoch_seconds else 7200.
                if time.time()-(current['progress'] or time.time())>limit:
                    # Only this task's identified child is terminated; checkpoint from
                    # the last complete epoch remains the exact resume boundary.
                    if alive(child.pid,current['start']):child.terminate()
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    q.fail(row['id'],'no scientific progress for %.0fs; last checkpoint retained'%limit,'transient');break
                time.sleep(10)
            if q.get(row['id'])['state']=='RUNNING':
                q.fail(row['id'],'child exited %s without a validated result'%child.returncode,'transient')
    q.close()

def supervisor(root):
    root=Path(root);handle=(root/'supervisor.lock').open('a')
    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    config=read(root/'runtime.json');q=Queue(root)
    write(root/'supervisor.json',{'pid':os.getpid(),'start':process_start(os.getpid()),'source_sha':code_sha(),'started_at':utc()})
    children={};logs=[];specs=[('cpu','cpu','')]+[('gpu'+str(i),'gpu',g) for i,g in enumerate(config['gpu_uuids'])]
    def spawn(name,args):
        log=(root/'logs'/(name+'.log')).open('ab',buffering=0);logs.append(log)
        p=subprocess.Popen([sys.executable,'-u','-B',str(ROOT/'scripts/qf_overnight/runtime.py')]+args+['--root',str(root)],
                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=os.environ.copy())
        children[name]=p
    while True:
        q.recover();q.refresh();terminal=all(r['state'] in TERMINAL for r in q.rows())
        stop=q.meta('stop',False)
        for name,resource,gpu in specs:
            if not stop and not terminal and (name not in children or children[name].poll() is not None):spawn(name,['worker','--resource',resource,'--gpu',gpu])
        if not stop and 'publisher' not in children or (not stop and children.get('publisher') and children['publisher'].poll() is not None):spawn('publisher',['publisher'])
        write(root/'processes.json',{name:{'pid':p.pid,'start':process_start(p.pid),'pgid':p.pid,'alive':p.poll() is None} for name,p in children.items()})
        write(root/'queue_status.json',q.snapshot())
        if terminal:
            # Publisher performs final snapshots and continues watching for peer
            # reports, so local completion never requires the other owner.
            q.setmeta('local_terminal',True)
            if q.meta('publisher_finished',False):break
        if stop and all(p.poll() is not None for name,p in children.items() if name!='publisher'):break
        time.sleep(10)
    q.close()

def initialize(args):
    root=Path(args.root);root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    if (root/'runtime.json').exists():
        current=read(root/'runtime.json');assert current['owner']==args.owner;print('Existing queue preserved:',root);return
    assert args.cache and args.publication and args.gpus,'cache, publication worktree and explicitly allocated GPU UUIDs required'
    ready=read(READY);assert ready['READY_CORE']
    from gates import runtime_gate
    runtime_gate(args.cache,ROOT)
    assert not subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain']).strip(),'execution worktree must be immutable and clean'
    config={'owner':args.owner,'source':str(ROOT),'source_sha':code_sha(),'python':sys.executable,'cache':str(Path(args.cache).resolve()),
            'publication':str(Path(args.publication).resolve()),'gpu_uuids':args.gpus.split(','),
            'result_branch':'results/qf-overnight-20261001-'+args.owner,'peer':'rtx3090' if args.owner=='a5000' else 'a5000',
            'gpu_mapping':[{'worker':i,'physical_uuid':u,'worker_cuda_visible_devices':u,'worker_device':'cuda:0'} for i,u in enumerate(args.gpus.split(','))],
            'publisher_seconds':300,'created_at':utc()}
    assert 0<len(config['gpu_uuids'])<=4 and len(set(config['gpu_uuids']))==len(config['gpu_uuids'])
    write(root/'runtime.json',config);q=Queue(root);q.initialize(read(CONFIG/'queue_manifest.json'),args.owner,ROOT);q.close();print(json.dumps(config))

def launch(root):
    root=Path(root);q=Queue(root);q.setmeta('stop',False)
    receipt=root/'supervisor.json'
    if receipt.exists():
        p=read(receipt)
        if alive(p['pid'],p['start']):print('Supervisor already running:',p['pid']);return
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
                                     CUBLAS_WORKSPACE_CONFIG=':4096:8',GIT_LFS_SKIP_SMUDGE='1',GIT_TERMINAL_PROMPT='0')
    with (root/'logs/supervisor.log').open('ab',buffering=0) as log:
        child=subprocess.Popen([sys.executable,'-u','-B',str(ROOT/'scripts/qf_overnight/runtime.py'),'supervisor','--root',str(root)],
                               stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    print(json.dumps({'supervisor_pid':child.pid,'process_group':child.pid,'detached':True,'logs':str(root/'logs')}))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['init','start','supervisor','worker','publisher','status','pause','resume','stop','retry','adapter-ready','adapter-blocked'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--owner',choices=['a5000','rtx3090']);p.add_argument('--cache');p.add_argument('--publication');p.add_argument('--gpus')
    p.add_argument('--resource',choices=['cpu','gpu']);p.add_argument('--gpu',default='');p.add_argument('--active',action='store_true');p.add_argument('--job');p.add_argument('--reason');p.add_argument('--source')
    a=p.parse_args()
    if a.action=='init':return initialize(a)
    if a.action=='start':return launch(a.root)
    if a.action=='supervisor':return supervisor(a.root)
    if a.action=='worker':return worker(a.root,a.resource,a.gpu)
    if a.action=='publisher':
        from publisher import loop
        return loop(a.root)
    q=Queue(a.root)
    if a.action=='status':print(json.dumps(q.snapshot(),indent=2))
    elif a.action=='pause':q.setmeta('paused',True);print('New claims paused; active jobs continue.')
    elif a.action=='resume':q.setmeta('paused',False);q.setmeta('stop',False);launch(a.root)
    elif a.action=='stop':
        q.setmeta('stop',True)
        if a.active:
            for name,record in read(a.root/'processes.json').items():
                if name!='publisher' and alive(record['pid'],record['start']):os.killpg(record['pgid'],signal.SIGTERM)
            print('Only identified task worker groups stopped; latest complete-epoch checkpoints preserved. Publisher flushes then exits.')
        else:print('New task claims stopped; active jobs finish before workers exit.')
    elif a.action=='retry':
        assert a.job and a.reason,'--job and explicit --reason required'
        row=q.get(a.job);assert row['state'] in {'BLOCKED','FAILED','CANCELLED'} and not alive(row['pid'],row['start'])
        assert row['retries']<2,'bounded retry budget exhausted; use a reviewed source/versioned invalidation, not endless retry'
        q.db.execute("UPDATE jobs SET state='PENDING',retries=retries+1,not_before=0,detail=? WHERE id=?",(json.dumps({'reviewed_retry_reason':a.reason}),a.job));q.event(a.job,'reviewed_retry',{'reason':a.reason});q.db.commit()
        print('Queued reviewed retry:',a.job)
    elif a.action=='adapter-ready':
        assert a.source;source=Path(a.source).resolve();r=read(source/'coordination/qf_overnight/20261001/ready.json')
        assert r['TLOB_ADAPTER_READY'] and r['protocol_hash']==q.meta('protocol_hash')
        q.adapter(source=source)
    elif a.action=='adapter-blocked':assert a.reason;q.adapter(reason=a.reason)
    q.close()

if __name__=='__main__':main()
