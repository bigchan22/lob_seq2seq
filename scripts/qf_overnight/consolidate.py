"""Incremental local/receiver aggregation; incomplete support is explicit, never zero."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from common import *
from scoring import losses,paired_bootstrap

def run(local,peer,owner):
    local=Path(local);peer=Path(peer);out=local/'aggregate';out.mkdir(exist_ok=True)
    indices=[]
    for root in [local,peer]:
        if (root/'export_manifest.json').exists():indices.append((root,read(root/'export_manifest.json')))
    records={};issues=[];fingerprints=[]
    for root,index in indices:
        for record in index['runs']:
            records[(index['owner'],record['job_id'])]=(root,record)
            fingerprints.append([index['owner'],record['job_id'],record.get('result_hash')])
    signature=digest(fingerprints)
    if (out/'analysis_state.json').exists() and read(out/'analysis_state.json').get('signature')==signature:return
    def resolve(role,id):
        item=records.get((role,id))
        if not item:return None
        root,record=item;result=record['result']
        if result.get('reused_from'):return resolve(role,result['reused_from'])
        path=root/record['relative_directory']/'predictions.csv.gz'
        if not path.is_file():return None
        return path,result
    metric_frames=[];asset_frames=[];subgroup_frames=[];calibration_frames=[]
    for root,index in indices:
        for record in index['runs']:
            if record['kind'] not in ['analysis','rolling_reuse','pool','prior','historical','intervention']:continue
            directory=root/record['relative_directory']
            for name,output in [('metrics.csv',metric_frames),('per_asset.csv',asset_frames),('subgroup_metrics.csv',subgroup_frames),('calibration.csv',calibration_frames)]:
                for path in directory.rglob(name):
                    frame=pd.read_csv(path);frame['owner']=index['owner'];frame['source_job']=record['job_id'];output.append(frame)
    for name,frames in [('metrics.csv',metric_frames),('per_asset.csv',asset_frames),('subgroup_metrics.csv',subgroup_frames),('calibration.csv',calibration_frames)]:
        if frames:
            combined=pd.concat(frames,ignore_index=True);combined.to_csv(out/name,index=False)
            if name=='metrics.csv':
                combined[combined.fold.ne('main')].to_csv(out/'rolling_metrics.csv',index=False)
                # Every 3-seed estimate is explicitly complete; never silently
                # change the scientific universe or average probabilities.
                keys=[k for k in ['model','fold','split','population','family'] if k in combined]
                summary=combined.groupby(keys,dropna=False).agg(seeds=('seed','nunique'),mean_seed_nll=('nll','mean'),mean_seed_accuracy=('accuracy','mean')).reset_index()
                summary['three_seed_complete']=summary.seeds.eq(3);summary.to_csv(out/'seed_summary.csv',index=False)
    differences=[];bootstrap=[]
    if owner=='rtx3090':
        for comparator in ['UC','UM_V2_DEPTH','UX','U1','GRU_U1','SHARED_QUERY','TLOB_ADAPTED']:
            for mode in ['main','pooled_rolling']:
                folds=['main'] if mode=='main' else ['R1','R2','R3']
                if mode!='main' and comparator not in ['UC','UM_V2_DEPTH']:continue
                left=[];right=[];refs=[];complete=True
                for fold in folds:
                    for seed in SEEDS:
                        def jid(model):return ('selected_'+model+'_42' if seed==42 else fit_name(model,seed)) if fold=='main' else ('R1_'+model+'_'+str(seed) if fold=='R1' else fit_name(model,seed,fold=fold))
                        aa=resolve('rtx3090',jid('UA'));bb=resolve(OWNERS[comparator],jid(comparator))
                        if not aa or not bb:complete=False;continue
                        pa,ra=aa;pb,rb=bb
                        assert ra['protocol_hash']==rb['protocol_hash'],'protocol mismatch: refuse combined scoring'
                        assert ra.get('training_hash')==rb.get('training_hash'),'train/validation/transform identity mismatch'
                        assert ra.get('executable_source_digest')==rb.get('executable_source_digest') or comparator=='TLOB_ADAPTED','core source mismatch'
                        a=pd.read_csv(pa,dtype={'date':str,'asset_id':str});b=pd.read_csv(pb,dtype={'date':str,'asset_id':str})
                        left.append(a);right.append(b);refs.append({'left_run':ra.get('run_id'),'right_run':rb.get('run_id'),'fold':fold,'seed':seed,
                                                                   'left_source_sha':ra.get('source_sha'),'right_source_sha':rb.get('source_sha')})
                if not complete:
                    issues.append({'comparison':'UA_minus_'+comparator,'mode':mode,'status':'incomplete','required_seeds':SEEDS,'required_folds':folds});continue
                a=pd.concat(left,ignore_index=True);b=pd.concat(right,ignore_index=True)
                wanted_splits=['validation','development_holdout'] if mode=='main' else ['rolling_evaluation']
                for split in wanted_splits:
                    for pop in POPS:
                        aa=a[a.split.eq(split)&a[pop].eq(1)].sort_values(KEYS+['seed']).reset_index(drop=True)
                        bb=b[b.split.eq(split)&b[pop].eq(1)].sort_values(KEYS+['seed']).reset_index(drop=True)
                        assert aa[KEYS+['seed','legacy_label']].equals(bb[KEYS+['seed','legacy_label']])
                        delta=losses(aa)[0]-losses(bb)[0]
                        differences.append(dict(comparison='UA_minus_'+comparator,mode=mode,split=split,population=pop,
                                                rows_including_seeds=len(aa),delta_nll=float(delta.mean()),seeds=3,status='complete'))
                        if comparator in ['UC','UM_V2_DEPTH']:
                            result=paired_bootstrap(aa,bb)
                            bootstrap.append(dict(comparison='UA_minus_'+comparator,mode=mode,split=split,population=pop,**result))
                write(out/('sources_UA_'+comparator+'_'+mode+'.json'),refs)
    if differences:pd.DataFrame(differences).to_csv(out/'paired_differences.csv',index=False)
    if bootstrap:write(out/'bootstrap_intervals.json',bootstrap)
    write(out/'bootstrap_settings.json',dict(draws=10000,block_dates=5,seed=20261001,whole_date_panels=True,fold_boundaries_preserved=True,row_weighted=True))
    write(out/'incomplete_comparisons.json',issues)
    write(out/'analysis_state.json',{'signature':signature,'updated_at':utc(),'owner':owner,'result_receipts':'See publisher peer_receipt.json and external local publication receipt for exact result refs; source commits included per run.'})
