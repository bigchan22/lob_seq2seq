"""Adapter for inspected producer p3-v1 READY/native queue; no model/trainer edits."""
from __future__ import annotations
import argparse,fcntl,gzip,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
from receiver import Receiver,atomic,read,sha,identity,alive,now,PREFIX

FROZEN_READY='e2212c40566e21c680a98186ebd8e984aa80d4fd'
FROZEN_PROTOCOL='bd1488a2d411295d7a2118c50ddcc0b158786123613af1676062e2270d1637d1'
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

class Native:
    def __init__(self,config):
        self.r=Receiver(config);self.root=self.r.root;self.c=self.r.c
        self.source=Path(self.c['source_root_parent'])/FROZEN_READY
        self.ready=read(self.source/'coordination/qf_overnight/20261001/ready.json')
        self.state=read(self.root/'native_status.json',{'phase':'VERIFYING_NATIVE_READY','ready_sha':FROZEN_READY,'protocol_hash':FROZEN_PROTOCOL})
        self.deadline=read(self.root/'receiver_status.json')['deadline_epoch']
        self.queue=self.root/'jobs';self.last_pub=0
    def save(self):
        self.state.update(updated_at=now(),pid=os.getpid(),start_ticks=identity(os.getpid()))
        atomic(self.root/'native_status.json',self.state)
        atomic(self.r.pub/PREFIX/'native_receiver_status.json',{k:v for k,v in self.state.items() if k not in ('pid','start_ticks')})
    def publish(self):
        # Before handoff only. The native publisher becomes sole Git writer after start.
        if (self.queue/'runtime.json').exists() and self.state.get('native_started'):return
        self.r.state.update(phase=self.state['phase'],ready_sha=FROZEN_READY,protocol_hash=FROZEN_PROTOCOL,
                            last_error=self.state.get('blocker'),native_receiver=True)
        self.r.save();self.r.publish();self.last_pub=time.time()
    def verify_source(self):
        assert self.r.git('rev-parse','HEAD',where=self.source).stdout.strip()==FROZEN_READY
        assert not self.r.git('status','--porcelain',where=self.source).stdout.strip()
        r=self.ready;p=read(self.source/'docs/qf_overnight/20261001/protocol.json')
        assert r['READY_CORE'] and digest(p)==r['protocol_hash']==FROZEN_PROTOCOL
        assert p['id']==self.c['run_group'] and p['data_handoff_sha']==self.c['data_sha']
        assert p['input_lineage']==self.c['input_sha']
        assert p['input_manifest_sha256']==r['input_manifest_sha256']==sha(self.source/'recovery/input_manifest.csv')
        for path,expected in r['core_source_files'].items():assert sha(self.source/path)==expected,path
        for entry in r.get('files',[]):assert sha(self.source/entry['path'])==entry['sha256'],entry['path']
        assert digest(r['core_source_files'])==r['core_executable_digest']
        files={'configs/qf_overnight/20261001/queue_manifest.json':r['manifest_sha256'],
               'configs/qf_overnight/20261001/splits.json':r['split_manifest_sha256'],
               'docs/qf_overnight/20261001/core_gates.json':r['core_gates_sha256']}
        for path,expected in files.items():assert sha(self.source/path)==expected,path
        splits=read(self.source/'configs/qf_overnight/20261001/splits.json');dates=p['dates']
        assert len(dates)==493 and dates==sorted(set(dates)) and len(p['assets'])==27
        for fold,bounds in {'main':[0,345,394,493],'R1':[0,345,394,427],'R2':[0,378,427,460],'R3':[0,411,460,493]}.items():
            a,t,v,e=bounds;s=splits[fold];assert s['indices_zero_based_half_open']==bounds
            assert s['train']==dates[a:t] and s['validation']==dates[t:v]
            assert s['development_holdout' if fold=='main' else 'rolling_evaluation']==dates[v:e]
        assert digest(splits)==p['split_manifest_hash']
        assert p['lrs']==[.00005,.0001,.0002,.0004] and p['selection_seed']==42 and p['confirmation_seeds']==[7,123]
        assert 'P3 only' in p['loss_mask'] and p['training']['effective_days']==32 and p['training']['microbatch']==8
        assert p['training']['max_epochs']==200 and p['training']['min_epochs']==20 and p['training']['patience']==30 and p['training']['min_delta']==.0001
        m=read(self.source/'configs/qf_overnight/20261001/queue_manifest.json');assert m['protocol_hash']==FROZEN_PROTOCOL
        jobs=m['jobs'];ids={j['job_id'] for j in jobs};assert len(ids)==len(jobs)==247
        own=[j for j in jobs if j['owner']=='rtx3090'];assert len(own)==69
        assert sum(j['kind']=='fit' for j in own)==30
        for j in jobs:
            x=dict(j);expected=x.pop('job_spec_hash');assert digest(x)==expected
            assert set(j['deps'])<=ids
        for j in own:assert set(j['deps'])<={x['job_id'] for x in own}
        for model,count in [('UA',12),('UC',12),('SHARED_QUERY',6)]:assert sum(j['kind']=='fit' and j['config'].get('model')==model for j in own)==count
        result={'verified':True,'ready_sha':FROZEN_READY,'protocol_hash':FROZEN_PROTOCOL,'core_executable_digest':r['core_executable_digest'],'global_dag_nodes':247,'owned_dag_nodes':69,'owned_new_fits':30,'source_files_verified':len(r['core_source_files']),'resolved_splits_verified':True,'historical_replay_sha':'dee398fb13078937833329a49f0665eb3c459b47'}
        atomic(self.root/'native_source_verification.json',result);atomic(self.r.pub/PREFIX/'native_source_verification.json',result)
    def cache(self):
        wanted=self.ready['prepared_data_hashes'];dest=self.queue/'prepared';dest.mkdir(parents=True,exist_ok=True)
        source=self.root/self.c.get('local_prepared_directory','prepared')
        original=read(source/'identity.json');received=read(self.root/'prepared_cache_receipt.json',{'files':[]})
        missing=[]
        for name,expected in wanted.items():
            target=dest/name
            if target.exists():assert sha(target)==expected;continue
            local=source/name
            if local.exists() and sha(local)==expected:
                with target.open('xb') as f:
                    with local.open('rb') as g:shutil.copyfileobj(g,f)
                received['files'].append({'name':name,'sha256':expected,'bytes':target.stat().st_size,'provenance':'local common prepare; exact frozen byte match'})
            else:missing.append(name)
        if missing:
            commit=self.r.advertised(self.c['shared_ref'])
            if not commit:return None
            tree=self.r.get_source(self.c['shared_ref'],commit)
            rr=json.loads(self.r.git('cat-file','blob',commit+':coordination/qf_overnight/20261001/ready.json').stdout)
            assert rr['protocol_hash']==FROZEN_PROTOCOL and rr['core_source_files']==self.ready['core_source_files'],'Cache export changed frozen core/protocol; needs versioned review'
            entries=[]
            for line in tree.splitlines():
                meta,path=line.split('\t');mode,kind,oid=meta.split()
                if kind=='blob' and mode!='120000' and 'qf_overnight/20261001/' in path and Path(path).name in missing+[x+'.gz' for x in missing]:entries.append((path,oid))
            for name in missing:
                candidates=[(p,o) for p,o in entries if Path(p).name in (name,name+'.gz')]
                for path,oid in candidates:
                    data=subprocess.run(['git','-C',str(self.r.bridge),'cat-file','blob',oid],env=self.r.env,capture_output=True,check=True,timeout=180).stdout
                    compressed_hash=hashlib.sha256(data).hexdigest();raw=gzip.decompress(data) if path.endswith('.gz') else data
                    if len(raw)>100_000_000 or hashlib.sha256(raw).hexdigest()!=wanted[name]:raise ValueError('Published canonical cache hash mismatch: '+path)
                    with (dest/name).open('xb') as f:f.write(raw)
                    received['files'].append({'name':name,'sha256':wanted[name],'bytes':len(raw),'source_commit':commit,'source_path':path,'git_blob_oid':oid,'transfer_sha256':compressed_hash});break
            missing=[name for name in wanted if not (dest/name).exists()]
        atomic(self.root/'prepared_cache_receipt.json',received)
        if missing:
            self.state.update(phase='WAITING_EXACT_PREPARED_CACHE',blocker='Missing Git-published frozen cache bytes: '+','.join(missing));return None
        original['files']=wanted
        target=dest/'identity.json'
        if not target.exists():target.write_text(json.dumps(original,indent=2,allow_nan=False)+'\n')
        assert sha(target)==self.ready['prepared_identity_sha256'],'Canonical identity mismatch'
        received.update(complete=True,identity_sha256=sha(target),ready_sha=FROZEN_READY)
        atomic(self.root/'prepared_cache_receipt.json',received);atomic(self.r.pub/PREFIX/'prepared_cache_receipt.json',received)
        return dest
    def invoke(self,args,log,env=None,timeout=120):
        with (self.root/log).open('ab') as f:
            p=subprocess.run([self.c['python'],'-u','-B',str(self.source/'scripts/qf_overnight/runtime.py')]+args,cwd=self.source,env=env or self.r.env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
        assert p.returncode==0,log+' failed; see local log'
    def launch(self,cache):
        ok,reason=self.r.resource_available()
        if not ok:self.state.update(phase='WAITING_FOR_ALLOCATED_GPU',blocker=reason);return False
        if self.state.get('deterministic_smoke_failed'):
            self.state.update(phase='BLOCKED_NATIVE_SMOKE',blocker='Recorded gate failure; preserve artifacts and await reviewed repair');return False
        # Producer bridge uses this exact task-local prepared path; no input patch.
        assert cache==self.queue/'prepared'
        sys.path.insert(0,str(self.source/'scripts/qf_overnight'))
        from queue_store import Queue
        q=Queue(self.queue);q.initialize(read(self.source/'configs/qf_overnight/20261001/queue_manifest.json'),'rtx3090',self.source);q.close()
        self.reuse_historical()
        self.state.update(phase='NATIVE_REAL_DATA_GATES',blocker=None);self.save()
        try:self.r.launch(self.ready,self.source)
        except Exception:
            self.state['deterministic_smoke_failed']=True;raise
        if self.r.state.get('phase')!='RUNNING':return False
        atomic(self.root/'pinned_ready.json',self.ready)
        self.r.state.update(ready_sha=FROZEN_READY,source_path=str(self.source),
                            protocol_hash=self.ready['protocol_sha256'],scientific_protocol_hash=FROZEN_PROTOCOL,
                            last_error=None,native_receiver=False)
        self.r.save();self.r.publish()
        entry=Path(__file__).with_name('receiver.py')
        with (self.root/'receiver.log').open('ab') as log:
            process=subprocess.Popen([sys.executable,'-u','-B',str(entry),'--config',self.r.config_path,'run'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=self.r.env)
        receipt={'script_path':str(entry),'pid':process.pid,'start_ticks':identity(process.pid),'process_group':process.pid,'started_at':now(),'native_bridge_source_sha':FROZEN_READY}
        atomic(self.root/'launch_receipt.json',receipt)
        self.state.update(phase='HANDED_TO_RUNNING_PRODUCER_BRIDGE',native_started=True,outer_receiver_pid=process.pid,
                          coordinator=self.r.state['coordinator'])
        atomic(self.root/'native_handoff.json',{'ready_sha':FROZEN_READY,'source_path':str(self.source),'python':self.c['python'],'queue_root':str(self.queue),'runtime_script':str(self.source/'scripts/qf_overnight/receiver_bridge/bridge.py'),'gpu_uuids':self.r.state['active_gpu_uuids'],'started_at':now()})
        self.save();return True
    def reuse_historical(self):
        # User permits reference/adoption of the exact completed replay instead of duplicate work.
        sys.path.insert(0,str(self.source/'scripts/qf_overnight'))
        from queue_store import Queue
        m=read(Path(self.c['data_worktree'])/'docs/qf_validity_replay/20261001/replay_manifest.json')
        idx=read(self.source/'docs/qf_data_review/20261001/prediction_index.json')
        assert m['status']=='complete' and m['data_handoff_sha']==self.c['data_sha']
        assert m['prediction_index_sha256']==sha(self.source/'docs/qf_data_review/20261001/prediction_index.json')
        assert {(x['model'],x['seed'],x['prediction_sha256']) for x in idx['selected']}=={(x['model'],x['seed'],x['prediction_sha256']) for x in m['prediction_records']}
        assert m['scoring']['dtype']=='float64' and m['scoring']['log']=='natural' and m['scoring']['weight']=='equal forecast rows'
        q=Queue(self.queue)
        if q.get('historical_replay')['state']=='PENDING':
            out=self.queue/'runs/historical_replay/attempt_0000';out.mkdir(parents=True,exist_ok=True)
            result={'validated':True,'historical_only':True,'reused_published_replay_sha':'dee398fb13078937833329a49f0665eb3c459b47','files':9,'replay_manifest_sha256':sha(Path(self.c['data_worktree'])/'docs/qf_validity_replay/20261001/replay_manifest.json'),'frozen_index_sha256':m['prediction_index_sha256'],'scoring_source_sha256':m['scoring_source_sha256'],'scoring_convention':m['scoring'],'source_sha':FROZEN_READY,'protocol_hash':FROZEN_PROTOCOL,'no_new_training_or_inference':True,'reuse_reason':'Complete independently verified historical replay; exact frozen artifacts and scoring convention; user explicitly permits published replay reference'}
            atomic(out/'result.json',result);q.success('historical_replay',out/'result.json')
        q.close()
    def run(self):
        handle=(self.root/'native_receiver.lock').open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.verify_source();delay=15
        while True:
            control=read(self.root/'native_control.json',{})
            if control.get('stop'):self.state['phase']='RECEIVER_STOPPED';self.save();return
            try:
                if self.state.get('native_started'):
                    receipt=read(self.queue/'supervisor.json');status=read(self.queue/'queue_status.json',{})
                    self.state.update(phase='RUNNING_NATIVE_QUEUE',queue_counts=status.get('counts'),native_supervisor=receipt)
                    if receipt and not alive({'pid':receipt['pid'],'start_ticks':receipt['start']}):
                        terminal=all(j['state'] in ('SUCCEEDED','BLOCKED','FAILED','CANCELLED') for j in status.get('jobs',[]))
                        self.state['phase']='NATIVE_QUEUE_TERMINAL' if terminal else 'NATIVE_SUPERVISOR_EXIT'
                        if not terminal and not status.get('stop') and self.state.get('supervisor_restarts',0)<2:
                            self.state['supervisor_restarts']=self.state.get('supervisor_restarts',0)+1
                            self.invoke(['start','--root',str(self.queue)],'native_restart.log')
                    self.save();time.sleep(60);continue
                if time.time()>self.deadline:
                    self.state.update(phase='BLOCKED_READY_OR_CACHE_TIMEOUT',blocker='No compatible executable inputs within original12-hour deadline');self.save()
                    self.r.state.update(self.state);self.r.final_publish();return
                if not control.get('paused'):
                    cache=self.cache()
                    if cache and self.launch(cache):return
                self.save()
                if time.time()-self.last_pub>300:self.publish()
                delay=min(120,delay*2)
            except Exception as error:
                self.state.update(phase='BLOCKED_NATIVE_VALIDATION',blocker=str(error));self.save()
                self.r.issue('NATIVE_VALIDATION',str(error),FROZEN_READY)
                try:self.publish()
                except Exception as e:self.state['publication_error']=str(e)
                delay=120
            time.sleep(delay)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('action',choices=['run','status','pause','resume','stop','retry']);p.add_argument('--active',action='store_true');p.add_argument('--job');p.add_argument('--reason');a=p.parse_args();n=Native(a.config)
    if a.action=='run':return n.run()
    if a.action=='status':
        print(json.dumps({'receiver':read(n.root/'native_status.json'),'native_queue':read(n.queue/'queue_status.json'),'publication':read(n.queue/'publication_status.json',read(n.root/'publication_receipt.json'))},indent=2));return
    ctl={'paused':a.action=='pause','stop':a.action=='stop','updated_at':now()};atomic(n.root/'native_control.json',ctl)
    if (n.queue/'runtime.json').exists():
        args=[a.action,'--root',str(n.queue)]
        if a.action=='stop' and a.active:args+=['--active']
        if a.action=='retry':
            if not a.job or not a.reason:p.error('retry requires exact --job and --reason')
            args+=['--job',a.job,'--reason',a.reason]
        n.invoke(args,'native_control.log')
    if a.action=='resume':
        record=read(n.root/'native_status.json',{})
        if not alive({'pid':record.get('pid',-1),'start_ticks':record.get('start_ticks')}):
            with (n.root/'native_receiver.log').open('ab') as log:
                subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__).resolve()),'--config',a.config,'run'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('Native task control:',a.action)
if __name__=='__main__':main()
