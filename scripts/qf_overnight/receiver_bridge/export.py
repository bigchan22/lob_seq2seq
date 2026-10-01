"""Transport projection only: exact saved probabilities, immutable selected exports.

The canonical scientific protocol hash and every actual source digest are retained.
The receiver's protocol_hash is the byte hash of the documented wire projection.
"""
import hashlib,json,shutil
from pathlib import Path
import pandas as pd
import numpy as np
from common import ROOT,READY,POPS,KEYS,read,write,sha,digest
from data import Data
from queue_store import Queue
from jobs import artifact,read_predictions
from scoring import losses,validate_predictions,metrics

def key_digest(frame):
    h=hashlib.sha256()
    for row in frame[['asset_id','date','origin_time','endpoint_time','split','legacy_label']].sort_values(['asset_id','date','origin_time'],kind='stable').itertuples(index=False,name=None):
        h.update(('\t'.join(map(str,row))+'\n').encode())
    return h.hexdigest()

def export_compatible(root,destination):
    root=Path(root);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    ready=read(READY);runtime=read(root/'runtime.json');q=Queue(root);entries=[]
    for row in q.rows():
        if row['state']!='SUCCEEDED' or not row['result']:continue
        job=json.loads(row['payload']);cfg=job['config'];kind=job['kind'];rid=row['id']
        panel=None
        if kind=='analysis' and cfg.get('model')!='S':
            source=cfg['source_job'];panel='tuned'
        elif kind=='rolling_reuse':source=rid;panel='tuned'
        elif kind=='pool':source=rid;panel='fixed_0001'
        elif rid.startswith('fixed_') and cfg.get('model') in ['U0','U1']:
            source=rid;panel='fixed_0001'
        else:continue
        directory,result=artifact(q,source)
        if not (directory/'predictions.csv.gz').exists():continue
        tag=rid+'__'+sha(directory/'result.json')[:16];out=destination/tag
        if not (out/'complete.json').exists():
            frame=read_predictions(directory/'predictions.csv.gz');fold=result.get('fold','main')
            data=Data(runtime['cache'],fold);expected=pd.concat([data.keys(s) for s in data.ids if s!='train'],ignore_index=True)
            validate_predictions(frame,expected)
            # Receiver uses development_holdout for the evaluation slice of each fold.
            frame['split']=frame.split.replace({'rolling_evaluation':'development_holdout'})
            base={'namespace':'new_p3_training','selected':True,'panel':panel,'model':cfg['model'],
                  'seed':cfg['seed'],'fold':fold,'protocol_hash':ready['protocol_sha256'],
                  'canonical_protocol_hash':q.meta('protocol_hash'),
                  'input_hash':data.identity['input_manifest_sha256'],'mask_hash':data.identity['files']['masks.npy'],
                  'split_hash':data.split_hash,'core_source_digest':ready['comparison_contract_digest'],
                  'source_sha':result.get('source_sha',runtime['source_sha']),
                  'source_digest':result.get('executable_source_digest',ready['core_executable_digest']),
                  'config_hash':result.get('config_hash',digest(cfg)),
                  'source_job':source,'prediction_sha256':sha(directory/'predictions.csv.gz')}
            out.mkdir(parents=True,exist_ok=True);summary=[];daily=[]
            for split,g in frame.groupby('split'):
                for short,pop in zip(['P0','P1','P2','P3'],POPS):
                    part=g[g[pop].eq(1)].copy();ident=dict(base,split=split,population=short)
                    summary.append(dict(ident,**metrics(part),row_key_label_digest=key_digest(part)))
                    part['loss']=losses(part)[0]
                    for date,dd in part.groupby('date'):
                        daily.append(dict(ident,date=date,rows=len(dd),loss_sum=float(dd.loss.sum()),row_key_label_digest=key_digest(dd)))
            pd.DataFrame(summary).to_csv(out/'metrics.csv',index=False)
            pd.DataFrame(daily).to_csv(out/'daily_loss_sums.csv',index=False)
            write(out/'provenance.json',dict(base,selected_epoch=result.get('selected_epoch'),
                transport_note='Byte-hashed protocol projection contains unchanged canonical protocol. Comparison digest covers model/data/trainer/label/factor implementations; exact executable digests retained separately.'))
            write(out/'complete.json',{'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
        entries.append({'job':rid,'path':tag,'complete_sha256':sha(out/'complete.json')})
    write(destination/'index.json',{'entries':entries,'protocol_hash':ready['protocol_sha256'],'canonical_protocol_hash':q.meta('protocol_hash')})
    q.close();return entries

def publication(root):
    """Receiver-owned Git process consumes these immutable files, never our DB."""
    root=Path(root);q=Queue(root);dest=root/'exports';dest.mkdir(exist_ok=True)
    export_compatible(root,dest/'compatible');selected=set()
    for r in q.rows():
        if r['state']=='SUCCEEDED' and json.loads(r['payload'])['kind']=='select':selected.add(read(r['result'])['selected_job'])
    for r in q.rows():
        if r['state']!='SUCCEEDED' or not r['result']:continue
        j=json.loads(r['payload']);result=read(r['result']);source=Path(r['result']).parent
        out=dest/'runs'/(r['id']+'__'+sha(r['result'])[:16])
        if (out/'complete.json').exists():continue
        out.mkdir(parents=True,exist_ok=True)
        pred=j['kind']!='fit' or j['config'].get('family')!='lr_search' or r['id'] in selected
        for path in source.rglob('*'):
            if not path.is_file() or path.suffix not in {'.csv','.json','.gz','.md','.svg'}:continue
            if path.suffix=='.gz' and not pred:continue
            rel=path.relative_to(source)
            if rel.name=='metrics.csv':rel=rel.with_name('diagnostic_metrics.csv')
            target=out/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        write(out/'complete.json',{'validated':True,'source_result_sha256':sha(r['result'])})
    # A selected search fit may have been exported before selection. Export its
    # original probabilities under a distinct immutable selected-input path.
    for job in selected:
        source,result=artifact(q,job);target=dest/'selected_inputs'/job/'predictions.csv.gz'
        if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/'predictions.csv.gz',target)
    status=q.snapshot();write(root/'queue_status.json',status)
    status_file=dest/'snapshots'/('status_'+digest(status)+'.json');write(status_file,status)
    entries=[{'path':str(p.relative_to(root)),'sha256':sha(p)} for p in sorted(dest.rglob('*')) if p.is_file()]
    write(root/'publication_manifest.json',{'files':entries});q.close()
