"""Single-writer result exporter, Git publisher, peer inbox and timed reports."""
import fcntl,json,os,shutil,subprocess,time,traceback
from pathlib import Path
from common import *
from queue_store import Queue,TERMINAL

ALLOWED={'.csv','.json','.gz','.svg','.md','.txt'}
def git(root,args,timeout=60):
    r=subprocess.run(['git','-C',str(root)]+args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=timeout,
                     env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_LFS_SKIP_SMUDGE='1'))
    if r.returncode:raise RuntimeError('git '+args[0]+' failed: '+r.stderr[-1200:])
    return r.stdout.strip()
def sanitized(text,runtime,root):
    for path,alias in [(str(root),'${TASK_ROOT}'),(runtime['source'],'${SOURCE_ROOT}'),(runtime['cache'],'${PREPARED_CACHE}')]:text=text.replace(path,alias)
    return text
def copy_file(source,dest):
    source=Path(source);dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and sha(dest)==sha(source):return
    if source.stat().st_size<=45_000_000:
        tmp=dest.with_suffix(dest.suffix+'.tmp');shutil.copyfile(source,tmp);os.replace(tmp,dest);return
    # Explicit lossless parts, never a misleading data pointer or Git-limit bypass.
    folder=dest.with_name(dest.name+'.parts');folder.mkdir(exist_ok=True);parts=[]
    with source.open('rb') as stream:
        i=0
        for block in iter(lambda:stream.read(16<<20),b''):
            path=folder/('part_%04d'%i);path.write_bytes(block);parts.append({'path':path.name,'bytes':len(block),'sha256':sha(path)});i+=1
    write(folder/'manifest.json',{'original_name':dest.name,'bytes':source.stat().st_size,'sha256':sha(source),'parts':parts,
                                'reassemble':'Concatenate listed parts in this exact order, then verify bytes and SHA256 before reading.'})
def export(root):
    root=Path(root);runtime=read(root/'runtime.json');pub=Path(runtime['publication']);owner=runtime['owner'];prefix='results/qf_overnight/20261001/'+owner
    # Never stage or overwrite files belonging to the other owner or code.
    status=git(pub,['status','--porcelain','--untracked-files=all'])
    for line in status.splitlines():assert line[3:].startswith(prefix+'/') or line[3:]==prefix+'/',line
    dest=pub/prefix;dest.mkdir(parents=True,exist_ok=True);q=Queue(root);snapshot=q.snapshot()
    snapshot.update(gpu_mapping=runtime['gpu_mapping'],source_sha=runtime['source_sha'],prepared_data_hashes=read(Path(runtime['cache'])/'identity.json')['files'])
    write(dest/'queue_status.json',json.loads(sanitized(json.dumps(snapshot),runtime,root)))
    rows=q.rows();by={r['id']:r for r in rows};selected={};selected_ids=set();trials=[]
    for row in rows:
        job=json.loads(row['payload'])
        if job['kind']=='select' and row['state']=='SUCCEEDED':
            result=read(row['result']);selected[result['model']]=result;selected_ids.add(result['selected_job'])
        if job['kind']=='fit' and job['config'].get('family')=='lr_search':
            trial={'job_id':row['id'],'model':job['config']['model'],'lr':job['config']['lr'],'state':row['state'],'validation_p3_nll':None}
            if row['state']=='SUCCEEDED':
                result=read(row['result']);trial.update(validation_p3_nll=result['validation_p3_nll'],selected_epoch=result['selected_epoch'],run_id=result['run_id'])
            elif row['state'] in ['FAILED','BLOCKED']:trial['failure']=json.loads(row['detail'] or '{}')
            trials.append(trial)
    write(dest/'selected_configs.json',{'criterion':'main P3 validation only; min_delta checkpoint; LR ties<=1e-8 smaller','selected':selected,'every_lr_trial':trials})
    exported=[]
    for row in rows:
        if row['state']!='SUCCEEDED' or not row['result']:continue
        job=json.loads(row['payload']);source=Path(row['result']).parent;result=read(row['result']);rel='runs/'+row['id']
        target=dest/rel;target.mkdir(parents=True,exist_ok=True)
        include_prediction=(job['kind']!='fit' or job['config'].get('family')!='lr_search' or row['id'] in selected_ids or
                            (job['config'].get('model') in ['U0','U1'] and job['config'].get('lr')==.0001))
        for path in source.rglob('*'):
            if not path.is_file() or path.suffix not in ALLOWED or path.name=='process.log':continue
            if path.suffix=='.gz' and not include_prediction:continue
            relative=path.relative_to(source)
            if relative.name=='metrics.csv':relative=relative.with_name('diagnostic_metrics.csv')
            copy_file(path,target/relative)
        # Earlier snapshots used this name for the non-interchange schema.
        # Remove only this redundant task-generated export, never run artifacts.
        for stale in target.rglob('metrics.csv'):
            assert stale.with_name('diagnostic_metrics.csv').exists()
            stale.unlink()
        if (source/'process.log').exists():
            text=(source/'process.log').read_text(errors='replace');(target/'log_tail.txt').write_text(sanitized('\n'.join(text.splitlines()[-60:])+'\n',runtime,root))
        exported.append({'job_id':row['id'],'kind':job['kind'],'relative_directory':rel,'result_hash':sha(source/'result.json'),'result':result,
                         'prediction_exported':include_prediction,'attempt':row['attempt']})
    manifest={'owner':owner,'run_group':GROUP,'protocol_hash':q.meta('protocol_hash'),'source_sha':runtime['source_sha'],'runs':exported}
    write(dest/'export_manifest.json',manifest)
    failures=[{'job':r['id'],'state':r['state'],'detail':json.loads(r['detail'] or '{}'),'source_sha':runtime['source_sha']} for r in rows if r['state'] in ['BLOCKED','FAILED']]
    write(dest/'issues.json',json.loads(sanitized(json.dumps(failures),runtime,root)))
    if (root/'issues').exists():
        for path in (root/'issues').rglob('*'):
            if path.is_file() and path.suffix in {'.json','.md','.py','.txt'}:copy_file(path,dest/'issues'/path.relative_to(root/'issues'))
    events=[dict(row) for row in q.db.execute('SELECT * FROM events ORDER BY seq')]
    (dest/'events.jsonl').write_text(''.join(sanitized(json.dumps(x),runtime,root)+'\n' for x in events))
    from consolidate import run
    run(dest,root/'peer_snapshot',owner)
    from receiver_bridge.export import export_compatible
    export_compatible(root,dest/'compatible')
    elapsed=(time.time()-q.meta('started_at'))/3600;counts=snapshot['counts'];terminal=all(r['state'] in TERMINAL for r in rows)
    completed=sum(r['state']=='SUCCEEDED' and json.loads(r['payload'])['kind'] in ['fit','fixed_or_alias'] and r['result'] and not read(r['result']).get('reused_from') for r in rows)
    running=[r for r in snapshot['jobs'] if r['state']=='RUNNING']
    report=f'''# {GROUP}: {owner} progress

Updated {utc()}; elapsed {elapsed:.2f} hours. Exact source commit `{runtime['source_sha']}`; protocol `{q.meta('protocol_hash')}`.
Queue states: `{json.dumps(counts,sort_keys=True)}`. Completed distinct full training jobs on this owner: {completed}; aliases and smokes are not counted as training. Local terminal: {terminal}.

Currently running: `{json.dumps([dict(job=r['id'],progress=r['detail']) for r in running])}`.

The finite global plan has153 unconditional full fits and up to4 conditional fixed-LR fits (157 maximum), plus explicit selection/reuse/analysis jobs. This owner never claims the other owner's models. A job succeeds only after result/prediction validation. Checkpoints remain task-local; hashes/relative names accompany each run. All selected compressed probabilities, histories, masks and scores are transferred through this result branch. Every LR trial and failure is recorded in selected_configs.json; performance never gates further valid work.

Read metrics and per-asset/subgroup/calibration tables under runs/ and aggregate/. Incomplete shared comparisons are explicitly listed in aggregate/incomplete_comparisons.json, not zero-filled. Means are per-seed losses, not probability ensembles. Historical replay is separate. R1 cites exact main-fit reuse; R2/R3 use their own training-only transforms. P0–P3 and validation/development_holdout/rolling_evaluation remain separate. Paired bootstrap uses10000 whole-date moving-block draws, length5, seed20261001, row weights and fold boundaries.

Limitations: provider aggregation/release timing unknown; historical selection and retrospective holdout; only3 optimization seeds; adapted-model scope where applicable. These classification results do not establish executable trading profitability or causality. P3 uses future endpoint quality ex post and never masks peer input. UC historical description conflict is preserved in the frozen replay namespace.

This report does not wait for the peer's final report to describe local completion. Local publication receipts outside Git and peer_receipt.json identify advertised result SHAs; no file tries to embed its own final commit SHA.
'''
    (dest/'progress_report.md').write_text(report)
    if terminal:(dest/'final_report.md').write_text(report+'\nLocal finite queue exhausted. Blocked/incomplete counts above remain explicit.\n')
    if elapsed>=8 and not (dest/'progress_8h.md').exists():(dest/'progress_8h.md').write_text(report+'\nScheduled8-hour snapshot; active jobs were not stopped.\n')
    write(root/'queue_status.json',snapshot);q.close();return pub,prefix,dest

def fetch_peer(root):
    root=Path(root);runtime=read(root/'runtime.json');pub=Path(runtime['publication']);peer=runtime['peer'];branch='results/qf-overnight-20261001-'+peer
    advertised=git(pub,['ls-remote','--heads','origin','refs/heads/'+branch],45)
    if not advertised:return {'available':False,'branch':branch}
    commit=advertised.split()[0];receipt=root/'peer_receipt.json'
    if receipt.exists() and read(receipt).get('sha')==commit:return read(receipt)
    ref='refs/remotes/origin/'+branch
    git(pub,['fetch','--filter=blob:none','--no-tags','--no-recurse-submodules','origin','refs/heads/'+branch+':'+ref],90)
    assert git(pub,['rev-parse',ref])==commit
    prefix='results/qf_overnight/20261001/'+peer+'/';names=git(pub,['ls-tree','-r','--name-only',commit,'--',prefix]).splitlines()
    destination=root/'peer_snapshot';destination.mkdir(exist_ok=True)
    for name in names:
        if not name.startswith(prefix):continue
        relative=name[len(prefix):]
        if runtime['owner']=='a5000' and name.endswith('.gz'):continue
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        # cat/show only task namespace; blob-filter fetch never materializes old history.
        result=subprocess.run(['git','-C',str(pub),'show',commit+':'+name],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90)
        if result.returncode:raise RuntimeError('peer task blob retrieval failed: '+relative)
        if not target.exists() or target.read_bytes()!=result.stdout:target.write_bytes(result.stdout)
    record={'available':True,'owner':peer,'branch':branch,'sha':commit,'verified_at':utc()};write(receipt,record);return record

def publish_once(root):
    root=Path(root);runtime=read(root/'runtime.json');pub=Path(runtime['publication']);owner=runtime['owner'];branch=runtime['result_branch']
    assert git(pub,['remote','get-url','origin'])==REMOTE
    assert git(pub,['branch','--show-current'])==branch
    peer_error=None
    try:peer=fetch_peer(root)
    except Exception as e:peer={'available':False,'error':str(e)};peer_error=str(e)
    pub,prefix,dest=export(root);write(dest/'peer_receipt.json',peer)
    git(pub,['add','--',prefix])
    staged=git(pub,['diff','--cached','--name-only']).splitlines();assert all(p.startswith(prefix+'/') for p in staged)
    git(pub,['diff','--cached','--check'])
    if staged:git(pub,['commit','-m','Update '+owner+' finite queue results and progress'],120)
    commit=git(pub,['rev-parse','HEAD']);error=None
    for delay in [0,5,20]:
        if delay:time.sleep(delay)
        try:
            git(pub,['push','origin','HEAD:refs/heads/'+branch],120)
            remote=git(pub,['ls-remote','--heads','origin','refs/heads/'+branch],45)
            assert remote.split()[0]==commit
            write(root/'publication_status.json',{'synced':True,'branch':branch,'sha':commit,'verified_at':utc(),'peer_error':peer_error});return
        except Exception as e:error=str(e)
    write(root/'publication_status.json',{'synced':False,'branch':branch,'local_sha':commit,'error':error,'updated_at':utc(),'next_retry':'next bounded publisher cycle'})

def loop(root):
    root=Path(root);handle=(root/'publisher.lock').open('a')
    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    last=0;last_terminal=False;last_phase=None;last_stop=False;terminal_since=None
    while True:
        q=Queue(root);rows=q.rows();terminal=all(r['state'] in TERMINAL for r in rows);stop=q.meta('stop',False)
        phase=min([json.loads(r['payload'])['priority'] for r in rows if r['state'] not in TERMINAL]+[999])
        elapsed=time.time()-q.meta('started_at');owner=q.meta('owner');q.close()
        if terminal and terminal_since is None:terminal_since=time.time()
        if time.time()-last>=300 or terminal!=last_terminal or phase!=last_phase or stop!=last_stop:
            try:publish_once(root)
            except Exception as error:
                traceback.print_exc();write(root/'publication_status.json',{'synced':False,'error':str(error),'updated_at':utc()})
            last=time.time();last_terminal=terminal;last_phase=phase;last_stop=stop
        if stop and not any(r['state']=='RUNNING' for r in rows):break
        if terminal and elapsed>=8*3600 and (root/'publication_status.json').exists() and read(root/'publication_status.json').get('synced'):
            peer_status=root/'peer_snapshot/queue_status.json'
            peer_value=read(peer_status) if peer_status.exists() else {}
            peer_value=peer_value.get('shared_queue_status',peer_value)
            peer_done=peer_value.get('all_terminal',False) or bool(peer_value.get('jobs')) and all(j['state'] in TERMINAL for j in peer_value['jobs'])
            peer_absent_timeout=not peer_status.exists() and time.time()-terminal_since>=24*3600
            if owner=='a5000' or peer_done or peer_absent_timeout:
                q=Queue(root);q.setmeta('publisher_finished',True);q.close();break
        time.sleep(10)
