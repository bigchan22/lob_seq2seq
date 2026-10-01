"""CPU consolidation of immutable, validated selected-result exports.

Does not define models, training masks or tuning. It checks exported support against
pinned producer rows; unsupported/missing exports stay explicitly incomplete.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

DATA_SHA='e85a89fd56584b513be940e6259c637369bd830b'
SEEDS=[42,7,123]
MODELS=['UA','UC','UM_V2_DEPTH','UX','U1','GRU_U1','SHARED_QUERY','U0','TLOB_ADAPTED']
ID=['model','seed','fold','split','population','panel']
HASHES=['protocol_hash','input_hash','split_hash','mask_hash','core_source_digest']
PROVENANCE=['source_sha','source_digest','config_hash']
POPS={'P0':'P0_legacy','P1':'P1_valid_cache_endpoints','P2':'P2_candidate_clock','P3':'P3_intersection'}

def dump(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');temp.replace(path)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(frame):
    h=hashlib.sha256()
    for row in frame[['asset_id','date','origin_time','endpoint_time','split','legacy_label']].sort_values(['asset_id','date','origin_time'],kind='stable').itertuples(index=False,name=None):
        h.update(('\t'.join(map(str,row))+'\n').encode())
    return h.hexdigest()
def expected_support(data,cache):
    path=data/'artifacts/qf_data_review/20261001/forecast_rows.csv.gz'
    key=sha(path)
    if cache.exists():
        value=json.loads(cache.read_text())
        if value['forecast_rows_sha256']==key and value['data_handoff_sha']==DATA_SHA:return value
    handoff=json.loads((data/'docs/qf_data_review/20261001/handoff.json').read_text())
    entries=handoff.get('files',[])
    # Receipt was verified before materialization; additionally pin the compressed payload here.
    checks=json.loads((data/'docs/qf_validity_replay/20261001/replay_manifest.json').read_text())
    assert checks['data_handoff_sha']==DATA_SHA and checks['status']=='complete'
    frame=pd.read_csv(path,keep_default_na=False,dtype={'asset_id':str,'date':str,'origin_time':str,'endpoint_time':str})
    assert digest(frame)==handoff['row_key_label_sha256']
    dates=sorted(frame.date.unique());assert len(dates)==493
    splits={'main':{'validation':dates[345:394],'development_holdout':dates[394:]},
            'R1':{'validation':dates[345:394],'development_holdout':dates[394:427]},
            'R2':{'validation':dates[378:427],'development_holdout':dates[427:460]},
            'R3':{'validation':dates[411:460],'development_holdout':dates[460:493]}}
    supports={}
    for fold,parts in splits.items():
        for split,ds in parts.items():
            base=frame[frame.date.isin(ds)].copy();base['split']=split
            for pop,col in POPS.items():
                f=base[base[col].eq(1)]
                daily={d:{'rows':len(g),'row_key_label_digest':digest(g)} for d in ds for g in [f[f.date.eq(d)]]}
                supports['|'.join([fold,split,pop])]={'rows':len(f),'dates':ds,'assets':f.asset_id.nunique(),'row_key_label_digest':digest(f),'daily':daily}
    value={'data_handoff_sha':DATA_SHA,'forecast_rows_sha256':key,'supports':supports,'digest_serialization':'sorted asset_id,date,origin_time; TAB six fields with endpoint,split,label; UTF8 LF no header'}
    dump(cache,value);return value

def check_pair(a,b):
    for key in HASHES+['row_key_label_digest','rows']:
        if str(a[key])!=str(b[key]):raise ValueError('Incompatible paired '+key)

def bootstrap(folds):
    rng=np.random.default_rng(20261001);ns=np.zeros(10000);ds=np.zeros(10000)
    for sums,counts in folds:
        n=len(sums);assert n>=5
        starts=rng.integers(0,n-4,size=(10000,(n+4)//5))
        ix=(starts[:,:,None]+np.arange(5)).reshape(10000,-1)[:,:n]
        ns+=sums[ix].sum(1);ds+=counts[ix].sum(1)
    assert (ds>0).all();return np.quantile(ns/ds,[.025,.975]).tolist()

def read_exports(root,owner,expected,protocol_hash):
    if root is None or not root.exists():return [],{},[owner+': published result snapshot unavailable']
    receipt=json.loads((root/'_receipt.json').read_text());commit=receipt['source_result_sha']
    for e in receipt['files']:
        if sha(root/e['path'])!=e['sha256']:raise ValueError('Snapshot file changed: '+e['path'])
    out=[];daily={};issues=[]
    metric_files=sorted(root.rglob('metrics.csv'))
    # Consolidated/historical copies must never become fresh source results.
    metric_files=[p for p in metric_files if 'consolidated' not in p.parts and 'historical' not in p.parts]
    if not metric_files:return [],{},[owner+': selected metrics.csv has not been published']
    for path in metric_files:
        frame=pd.read_csv(path,dtype=str,keep_default_na=False)
        required=set(ID+HASHES+PROVENANCE+['namespace','selected','rows','row_key_label_digest','nll'])
        if not required.issubset(frame.columns):
            issues.append(owner+': '+str(path.relative_to(root))+' missing columns '+repr(sorted(required-set(frame.columns))));continue
        dp=path.with_name('daily_loss_sums.csv')
        if not dp.exists():issues.append(owner+': '+str(dp.relative_to(root))+' missing');continue
        df=pd.read_csv(dp,dtype=str,keep_default_na=False)
        if not set(ID+['date','rows','loss_sum','row_key_label_digest']).issubset(df.columns):issues.append(owner+': invalid daily-loss schema');continue
        for rec in frame.to_dict('records'):
            if rec['selected'].lower() not in ('true','1'):continue
            if rec['namespace']!='new_p3_training':continue
            try:
                assert protocol_hash and rec['protocol_hash']==protocol_hash, 'pinned protocol mismatch'
                assert rec['input_hash']=='9cb89005693b1b70c102178849b740fde00b50c9ba5f3b5530d0e5749ea59837', 'input manifest mismatch'
                assert rec['panel'] in ['tuned','fixed_0001']
                assert rec['seed'] in ['42','7','123']
                assert rec['model'] in MODELS+['S']
                assert rec['population'] in POPS
                s=expected['supports']['|'.join(rec[k] for k in ['fold','split','population'])]
                assert int(rec['rows'])==s['rows'] and rec['row_key_label_digest']==s['row_key_label_digest']
                for k in HASHES+PROVENANCE:
                    assert len(rec[k]) in (40,64) and all(c in '0123456789abcdef' for c in rec[k]),k
                d=df.copy()
                for k in ID:d=d[d[k].eq(rec[k])]
                assert not d.date.duplicated().any() and set(d.date)==set(s['dates'])
                d=d.set_index('date').loc[s['dates']]
                sums=d.loss_sum.to_numpy(float);counts=d.rows.to_numpy(int)
                assert np.isfinite(sums).all() and (sums>=0).all()
                for i,date in enumerate(s['dates']):
                    assert counts[i]==s['daily'][date]['rows']
                    assert d.iloc[i].row_key_label_digest==s['daily'][date]['row_key_label_digest']
                assert abs(sums.sum()/counts.sum()-float(rec['nll']))<1e-9
                key=tuple(rec[k] for k in ID)
                if key in daily:raise ValueError('duplicate selected identity')
                daily[key]=(sums,counts)
                out.append({**rec,'owner':owner,'source_result_sha':commit,'metric_source':str(path.relative_to(root))})
            except (AssertionError,ValueError,KeyError) as error:
                issues.append(owner+': rejected '+repr(tuple(rec.get(k) for k in ID))+': '+str(error))
    return out,daily,issues

def write_csv(path,rows,empty_fields):
    fields=list(dict.fromkeys(k for r in rows for k in r)) or empty_fields
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def consolidate(data,local,other,out,cache,protocol_hash=None):
    out.mkdir(parents=True,exist_ok=True);expected=expected_support(data,cache)
    records=[];daily={};issues=[];sources={}
    for owner,root in [('rtx3090',local),('a5000',other)]:
        if root is not None and (root/'_receipt.json').exists():sources[owner]=json.loads((root/'_receipt.json').read_text())['source_result_sha']
        rr,dd,ii=read_exports(root,owner,expected,protocol_hash);records+=rr;issues+=ii
        for k,v in dd.items():
            if k in daily:raise ValueError('Same scientific identity exported by both owners: '+str(k))
            daily[k]=v
    index={tuple(r[k] for k in ID):r for r in records}
    if len(index)!=len(records):raise ValueError('Duplicate selected result identity')
    pairs=[];missing=[];roll={}
    comparisons=[('UA',m,'tuned') for m in ['UC','UM_V2_DEPTH','UX','U1','GRU_U1','SHARED_QUERY','TLOB_ADAPTED']]+[('S','U0','fixed_0001'),('U0','U1','fixed_0001')]
    for ma,mb,panel in comparisons:
        folds=['main','R1','R2','R3'] if (ma,mb) in [('UA','UC'),('UA','UM_V2_DEPTH')] else ['main']
        for fold in folds:
            for split in ['validation','development_holdout']:
                for pop in POPS:
                    vectors=[];pair_recs=[];valid=True
                    for seed in SEEDS:
                        ka=(ma,str(seed),fold,split,pop,panel);kb=(mb,str(seed),fold,split,pop,panel)
                        absent=[k for k in [ka,kb] if k not in index]
                        if absent:
                            missing.append({'model_a':ma,'model_b':mb,'seed':seed,'fold':fold,'split':split,'population':pop,'panel':panel,'status':'incomplete','missing':repr(absent)})
                            valid=False;continue
                        try:
                            a,b=index[ka],index[kb];check_pair(a,b)
                            aa,ac=daily[ka];bb,bc=daily[kb];assert np.array_equal(ac,bc)
                            d=aa-bb;vectors.append((d,ac));pair_recs.append((a,b))
                            pairs.append({'model_a':ma,'model_b':mb,'seed':seed,'fold':fold,'split':split,'population':pop,'panel':panel,'rows':int(ac.sum()),'delta_nll':float(d.sum()/ac.sum()),'status':'complete_per_seed','source_a':a['source_result_sha'],'source_b':b['source_result_sha'],'protocol_hash':a['protocol_hash']})
                        except (ValueError,AssertionError) as error:
                            issues.append('Pair rejected '+repr(ka)+': '+str(error));valid=False
                    if valid and len(vectors)==3:
                        for a,b in pair_recs[1:]:check_pair(pair_recs[0][0],a)
                        sums=np.mean([v[0] for v in vectors],axis=0);counts=vectors[0][1]
                        rec={'model_a':ma,'model_b':mb,'seed':'mean_3','fold':fold,'split':split,'population':pop,'panel':panel,'rows':int(counts.sum()),'delta_nll':float(sums.sum()/counts.sum()),'status':'complete_matched_3_seeds','source_a':pair_recs[0][0]['source_result_sha'],'source_b':pair_recs[0][1]['source_result_sha'],'protocol_hash':pair_recs[0][0]['protocol_hash']}
                        if (ma,mb) in [('UA','UC'),('UA','UM_V2_DEPTH')]:
                            rec['ci_lower'],rec['ci_upper']=bootstrap([(sums,counts)])
                            if fold in ['R1','R2','R3'] and split=='development_holdout':roll.setdefault((ma,mb,pop),{})[fold]=(sums,counts,rec)
                        pairs.append(rec)
    rolling=[]
    for key,folds in roll.items():
        if set(folds)!=set(['R1','R2','R3']):continue
        vectors=[folds[f][:2] for f in ['R1','R2','R3']];lo,hi=bootstrap(vectors)
        rolling.append({'model_a':key[0],'model_b':key[1],'population':key[2],'seed':'mean_3','rows':int(sum(c.sum() for s,c in vectors)),'delta_nll':float(sum(s.sum() for s,c in vectors)/sum(c.sum() for s,c in vectors)),'ci_lower':lo,'ci_upper':hi,'block_boundaries':'resample separately within R1/R2/R3','source_results':json.dumps(sources,sort_keys=True)})
    write_csv(out/'metrics.csv',records,ID+['rows','nll','source_result_sha'])
    write_csv(out/'paired_differences.csv',pairs,['model_a','model_b','fold','split','population','seed','status','delta_nll'])
    write_csv(out/'rolling_metrics.csv',rolling,['model_a','model_b','population','rows','delta_nll'])
    write_csv(out/'incomplete_comparisons.csv',missing,['model_a','model_b','seed','fold','split','population','status','missing'])
    manifest={'status':'partial' if missing or issues else 'complete','data_handoff_sha':DATA_SHA,'source_result_shas':sources,'validated_metric_rows':len(records),'complete_pair_rows':len(pairs),'incomplete_seed_comparisons':len(missing),'issues':issues,'bootstrap':{'replicates':10000,'block_length':5,'seed':20261001,'paired_whole_dates':True,'rolling_blocks_do_not_cross_folds':True},'namespace':'new_p3_training','limitations':'Daily sums and key/label digests originate in owner-validated saved-prediction exports. No probability ensemble; no inference or selection here. Intervals retrospective/conditional on selection. Missing results are not zero.'}
    manifest['checksums']=[{'path':p.name,'sha256':sha(p)} for p in sorted(out.glob('*.csv'))]
    dump(out/'consolidation_manifest.json',manifest)
    report='# Matched all-server analysis\n\nStatus: '+manifest['status']+'\n\nSource result commits: '+json.dumps(sources,sort_keys=True)+'\n\nValidated selected metric rows: '+str(len(records))+'; complete paired rows: '+str(len(pairs))+'; incomplete seed comparisons: '+str(len(missing))+'.\n\n'
    report+='No missing models/assets/seeds are inner-joined away or recorded as zero. Historical replay is separate. Table exports must match pinned per-date key/label digests, population counts and scientific hashes before pairing.\n\n'
    report+='Issues: '+json.dumps(issues)+'\n\n'+manifest['limitations']+'\n'
    (out/'progress_report.md').write_text(report)
    return manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--local',type=Path);p.add_argument('--other',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--protocol-hash');a=p.parse_args()
    print(json.dumps(consolidate(a.data,a.local,a.other,a.output,a.cache,a.protocol_hash),indent=2))
if __name__=='__main__':main()
