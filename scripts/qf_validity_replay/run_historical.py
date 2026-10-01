"""Independent CPU verification and historical saved-probability replay."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from replay_core import labels_from_raw_aligned_prices, moving_block_interval

DATA_SHA='e85a89fd56584b513be940e6259c637369bd830b'
KEYS=['asset_id','date','origin_time']
POPS=['P0_legacy','P1_valid_cache_endpoints','P2_candidate_clock','P3_intersection']
SEEDS=[42,7,123]
MODELS=['UA','UC','UM_V2_DEPTH']
EXPECTED={'all':[505818,469391,439263,432560],
          'train':[353970,323955,307395,302922],
          'validation':[50274,48280,43659,43050],
          'development_holdout':[101574,97156,88209,86588]}

def log(text): print(text,flush=True)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def digest(frame,columns):
    h=hashlib.sha256()
    for row in frame[columns].itertuples(index=False,name=None):
        h.update(('\t'.join(str(v) for v in row)+'\n').encode())
    return h.hexdigest()
def dump(path,obj): Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')

def reconstruct(root,inputs,contract,provided):
    """Read original cache values independently; never use provided flags to build flags."""
    manifest=pd.read_csv(root/'recovery/input_manifest.csv',keep_default_na=False)
    for row in manifest.itertuples(index=False):
        path=inputs/row.source_path
        assert path.stat().st_size==row.bytes and sha(path)==row.sha256,row.source_path
    grid=contract['grid']; frames={}; date_sets=[]
    qty=[f'{side}_STEP{i}_BSTORD_RQTY' for i in range(1,6) for side in ['ASK','BID']]
    ask='ASK_STEP1_BSTORD_PRC';bid='BID_STEP1_BSTORD_PRC'
    for record in manifest[manifest.category.eq('existing_runtime_csv_cache')].itertuples(index=False):
        asset=Path(record.source_path).stem
        df=pd.read_csv(inputs/record.source_path,usecols=['ORD_DD','TIME_INTERVAL',ask,bid]+qty,
                       dtype={'ORD_DD':str,'TIME_INTERVAL':str})
        assert df.ORD_DD.str.fullmatch(r'\d{8}').all()
        df['TIME_INTERVAL']=df.TIME_INTERVAL.str.strip()
        frames[asset]=df;date_sets.append(set(df.ORD_DD))
    dates=sorted(set.intersection(*date_sets));assets=sorted(frames)
    assert len(dates)==493 and len(assets)==27
    split_file=pd.read_csv(root/'docs/qf_data_review/20261001/split_dates.csv',dtype=str)
    assert split_file.date.tolist()==dates
    splits=['train']*345+['validation']*49+['development_holdout']*99
    assert split_file.split.tolist()==splits
    date_iso=pd.to_datetime(pd.Series(dates),format='%Y%m%d').dt.strftime('%Y-%m-%d').to_numpy()
    ix=pd.MultiIndex.from_product([dates,grid],names=['ORD_DD','TIME_INTERVAL'])
    pieces=[]
    for asset in assets:
        df=frames[asset];df=df[df.ORD_DD.isin(dates)&df.TIME_INTERVAL.isin(grid)]
        assert not df.duplicated(['ORD_DD','TIME_INTERVAL']).any(),asset
        orig=df.assign(_exists=1).set_index(['ORD_DD','TIME_INTERVAL']).reindex(ix)
        a=orig[ask].to_numpy().reshape(493,39);b=orig[bid].to_numpy().reshape(493,39)
        labels,mids=labels_from_raw_aligned_prices(b,a)
        exists=orig._exists.fillna(0).to_numpy().reshape(493,39).astype(np.int8)
        valid=np.isfinite(a)&np.isfinite(b)&(a>0)&(b>0)&(b<=a)
        finite=np.isfinite(a)&np.isfinite(b)
        piece=pd.DataFrame({'asset_id':asset,'date':np.repeat(date_iso,38),
                            'origin_time':np.tile(grid[:-1],493),'endpoint_time':np.tile(grid[1:],493),
                            'split':np.repeat(splits,38),'legacy_label':labels.ravel()})
        for prefix,sl in [('origin',slice(None,-1)),('endpoint',slice(1,None))]:
            piece[prefix+'_cache_row_exists']=exists[:,sl].ravel()
            piece[prefix+'_valid_original_cache_quote']=valid[:,sl].ravel().astype(np.int8)
            for name,arr in [('ask1',a),('bid1',b)]:
                vals=arr[:,sl].ravel()
                for kind,flag in [('present',~np.isnan(vals)),('finite',np.isfinite(vals)),
                                  ('finite_positive',np.isfinite(vals)&(vals>0)),
                                  ('nonpositive',np.isfinite(vals)&(vals<=0))]:
                    piece[prefix+'_cache_'+name+'_'+kind]=flag.astype(np.int8)
            piece[prefix+'_crossed']=(finite[:,sl]&(b[:,sl]>a[:,sl])).ravel().astype(np.int8)
            piece[prefix+'_locked_positive']=(finite[:,sl]&(a[:,sl]>0)&(b[:,sl]>0)&(b[:,sl]==a[:,sl])).ravel().astype(np.int8)
        present=~np.isnan(a)&~np.isnan(b)
        piece['missing_cache_quote_any_endpoint']=(~(present[:,:-1]&present[:,1:])).ravel().astype(np.int8)
        missing=(~(finite[:,:-1]&finite[:,1:])).ravel()
        equal=(mids[:,:-1]==mids[:,1:]).ravel()
        piece['historical_missing_cache_endpoint']=missing.astype(np.int8)
        piece['equal_filled_mids']=equal.astype(np.int8)
        piece['historical_missing_equal_filled_flat']=(missing&equal&(labels.ravel()==1)).astype(np.int8)
        p1=(valid[:,:-1]&valid[:,1:]).ravel()
        clock=((piece.origin_time>='09:10:00')&(piece.endpoint_time<='14:40:00')).to_numpy()
        piece[POPS[0]]=1;piece[POPS[1]]=p1.astype(np.int8);piece[POPS[2]]=clock.astype(np.int8);piece[POPS[3]]=(p1&clock).astype(np.int8)
        piece['no_cross_day_target']=1;piece['exact_ten_minute_target']=1
        origin_ts=pd.to_datetime(piece.date+' '+piece.origin_time)
        endpoint_ts=pd.to_datetime(piece.date+' '+piece.endpoint_time)
        assert (endpoint_ts-origin_ts).eq(pd.Timedelta(minutes=10)).all()
        # Analysis covariates use only legacy-filled CURRENT values, never the endpoint.
        fill=pd.DataFrame(orig[[ask,bid]+qty].to_numpy(),index=ix).groupby(level=0,sort=False).ffill().fillna(0).to_numpy(np.float32).reshape(493,39,12)
        aa=fill[:,:-1,0].astype(np.float64);bb=fill[:,:-1,1].astype(np.float64);mm=mids[:,:-1].astype(np.float64)
        good=np.isfinite(aa)&np.isfinite(bb)&(aa>0)&(bb>0)&(bb<=aa)
        spread=np.full(mm.shape,np.nan);np.divide(aa-bb,mm,out=spread,where=good)
        depth=fill[:,:-1,2:].astype(np.float64);dg=np.isfinite(depth).all(2)&(depth>=0).all(2)
        depth_sum=depth.sum(2);depth_sum[~dg]=np.nan
        piece['current_relative_spread']=spread.ravel();piece['current_l5_depth']=depth_sum.ravel()
        pieces.append(piece);log('cache flags/float32 labels: '+asset)
    actual=pd.concat(pieces,ignore_index=True)
    assert not actual.duplicated(KEYS).any()
    assert actual[KEYS].equals(provided[KEYS])
    cols=[c for c in actual if c not in ['current_relative_spread','current_l5_depth']]
    mismatches={c:int(np.count_nonzero(actual[c].to_numpy()!=provided[c].to_numpy())) for c in cols}
    assert not any(mismatches.values()),mismatches
    assert int(actual.historical_missing_cache_endpoint.sum())==34600
    assert int(actual.historical_missing_equal_filled_flat.sum())==27366
    for split,expect in EXPECTED.items():
        subset=actual if split=='all' else actual[actual.split.eq(split)]
        assert subset[POPS].sum().tolist()==expect,(split,subset[POPS].sum().tolist())
    return actual,{'rows':len(actual),'assets':len(assets),'common_dates':len(dates),
                  'compared_columns':cols,'per_column_mismatches':mismatches,
                  'original_cache_before_fill_verified':True,'missing_only_rows':34600,
                  'missing_equal_filled_flat_rows':27366,'input_files_hash_verified':54,
                  'no_full_workbook_comparison_repeated':True}

def metric(y,p,loss):
    pred=p.argmax(1);one=np.eye(3)[y]
    return {'nll':float(loss.mean()),'accuracy':float(accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),
            'mcc':float(matthews_corrcoef(y,pred)),
            'brier':float(np.mean(np.sum((p-one)**2,axis=1)))}
def coverage(rows,base_count):
    n=len(rows);counts=np.bincount(rows.legacy_label.to_numpy(),minlength=3)
    d={'rows':n,'dates':rows.date.nunique(),'assets':rows.asset_id.nunique(),'retained_fraction':n/base_count}
    for i in range(3):d[f'class_{i}_count']=int(counts[i]);d[f'class_{i}_proportion']=float(counts[i]/n) if n else None
    return d

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repository',type=Path,required=True);ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();root=a.repository.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'replay_manifest.json').exists():raise FileExistsError('Preserve completed replay; use its hashes or a fresh output directory')
    docs=root/'docs/qf_data_review/20261001';h=json.loads((docs/'handoff.json').read_text());c=json.loads((docs/'data_contract.json').read_text());idx=json.loads((docs/'prediction_index.json').read_text())
    assert sha(docs/'prediction_index.json')==h['selected_prediction_index_sha256']
    assert idx['frozen_before_restricted_scoring'] and len(idx['selected'])==9
    assert {(x['model'],x['seed']) for x in idx['selected']}=={(m,s) for m in MODELS for s in SEEDS}
    provided=pd.read_csv(root/h['forecast_rows'],dtype={k:str for k in KEYS+['endpoint_time','split']},keep_default_na=False)
    assert len(provided)==505818 and not provided.duplicated(KEYS).any()
    assert provided[KEYS].equals(provided.sort_values(KEYS,kind='stable')[KEYS])
    key_hash=digest(provided,KEYS);label_hash=digest(provided,KEYS+['endpoint_time','split','legacy_label'])
    assert key_hash==h['row_key_sha256'] and label_hash==h['row_key_label_sha256']
    rows,checks=reconstruct(root,a.inputs,c,provided)
    for split,expected_digest in h['split_digests'].items():
        assert digest(rows[rows.split.eq(split)],KEYS+['endpoint_time','split','legacy_label'])==expected_digest
    checks.update(row_key_sha256=key_hash,row_key_label_sha256=label_hash,mask_digest=digest(rows,KEYS+POPS))
    dump(out/'data_mask_checks.json',checks)
    train=rows.split.eq('train')&rows[POPS[3]].eq(1);cutpoints={}
    for cov in ['current_relative_spread','current_l5_depth']:
        val=rows[cov].to_numpy();good=np.isfinite(val);cuts=np.quantile(val[good&train.to_numpy()],[1/3,2/3]);cutpoints[cov]={'values':cuts.tolist(),'fit_rows':int((good&train.to_numpy()).sum()),'fit_population':'train P3 only','invalid_group_separate':True}
        group=np.full(len(rows),'invalid',dtype=object);group[good]=np.array(['low','middle','high'])[np.searchsorted(cuts,val[good],side='right')];rows[cov+'_bin']=group
    rows['origin_hour']=rows.origin_time.str[:2]+':00'
    dump(out/'train_only_subgroup_thresholds.json',cutpoints)
    counts=[]
    for split in ['train','validation','development_holdout']:
        base=rows[rows.split.eq(split)]
        for pop in POPS:
            mask=base[pop].eq(1);part=base[mask];record={'population':pop,'split':split,**coverage(part,len(base)),'excluded_total':int((~mask).sum())}
            # Overlapping validity reasons are diagnostics, not additive counts.
            scope=provided[provided.split.eq(split)]
            record['reason_counts_overlap']=True
            record['invalid_origin_quote']=int(scope.origin_valid_original_cache_quote.eq(0).sum())
            record['invalid_endpoint_quote']=int(scope.endpoint_valid_original_cache_quote.eq(0).sum())
            record['historical_missing_only']=int(scope.historical_missing_cache_endpoint.sum())
            record['outside_clock']=int(scope.P2_candidate_clock.eq(0).sum())
            counts.append(record)
    pd.DataFrame(counts).to_csv(out/'population_counts.csv',index=False)
    ev=rows[rows.split.ne('train')].reset_index(drop=True);index_expected=pd.MultiIndex.from_frame(ev[KEYS]);y=ev.legacy_label.to_numpy(dtype=np.int64)
    probabilities={};losses={};audits=[];reconciliations=[]
    references={('UA','validation'):.887593050,('UC','validation'):.903287486,('UA','development_holdout'):.908952596,('UC','development_holdout'):.922755800}
    for record in idx['selected']:
        model,seed=record['model'],record['seed'];path=root/record['prediction_path'];assert sha(path)==record['prediction_sha256'];cfg=path.parent/'resolved_config.json';assert sha(cfg)==record['configuration_sha256'];assert json.loads(cfg.read_text())==record['configuration']
        history=pd.read_csv(path.parent/'training_history.csv');best=float('inf');epoch=None
        for item in history.itertuples(index=False):
            if item.validation_log_loss<best-record['configuration']['min_delta']:best=float(item.validation_log_loss);epoch=int(item.epoch)
        assert epoch==record['selected_epoch'],(model,seed,epoch,record['selected_epoch'])
        pred=pd.read_csv(path,dtype={'asset':str,'date':str,'timestamp':str})
        assert pred.date.str.fullmatch(r'\d{8}').all();pred=pred.rename(columns={'asset':'asset_id','timestamp':'origin_time'})
        pred['date']=pd.to_datetime(pred.date,format='%Y%m%d').dt.strftime('%Y-%m-%d')
        assert pred.seed.eq(seed).all() and pred.model_id.eq(model).all()
        assert pred.validation_selected_checkpoint_epoch.eq(epoch).all()
        pi=pd.MultiIndex.from_frame(pred[KEYS]);assert not pi.has_duplicates
        assert len(index_expected.difference(pi))==len(pi.difference(index_expected))==0
        pred=pred.iloc[pi.get_indexer(index_expected)].reset_index(drop=True)
        assert pred.split.equals(ev.split);assert np.array_equal(pred.true_class,y)
        p=pred[record['probability_columns']].to_numpy(dtype=np.float64)
        assert record['probability_columns']==['probability_down','probability_flat','probability_up']
        assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all();sums=p.sum(1);assert (sums>0).all()
        err=np.abs(sums-1);assert err.max()<1e-5
        p=p/sums[:,None];loss=-np.log(np.maximum(p[np.arange(len(y)),y],np.finfo(np.float64).eps))
        probabilities[(model,seed)]=p;losses[(model,seed)]=loss
        for split in ['validation','development_holdout']:
            actual=float(loss[ev.split.eq(split)].mean())
            if seed==42 and (model,split) in references:
                target=references[(model,split)];difference=actual-target
                reconciliations.append({'model':model,'seed':seed,'split':split,'recorded_reference':target,'actual':actual,'difference':difference,'tolerance':1e-7,'pass':abs(difference)<=1e-7})
        audits.append({**record,'selection_replayed_epoch':epoch,'selected_history_validation_nll':best,'rows_matched':len(pred),'duplicates':0,'missing':0,'extra':0,'label_mismatches':0,'split_mismatches':0,'max_row_sum_error':float(err.max()),'mean_row_sum_error':float(err.mean())})
        log('verified frozen predictions: '+model+' seed '+str(seed))
    dump(out/'legacy_reconciliation.json',reconciliations);dump(out/'prediction_audit.json',audits)
    if not all(x['pass'] for x in reconciliations):raise ValueError('Historical anchor mismatch; do not interpret masked scores. New experiments remain independently runnable.')
    metrics=[];calibration=[];subgroups=[];per_asset_metrics=[]
    for (model,seed),p in probabilities.items():
        loss=losses[(model,seed)]
        for split in ['validation','development_holdout']:
            base=ev.split.eq(split).to_numpy()
            for pop in POPS:
                mask=base&ev[pop].eq(1).to_numpy();part=ev[mask];pp=p[mask];yy=y[mask];ll=loss[mask]
                asset_means=pd.DataFrame({'asset_id':part.asset_id.to_numpy(),'loss':ll}).groupby('asset_id').loss.mean()
                rec={'namespace':'historical_unmasked_training','model':model,'seed':seed,'population':pop,'split':split,**coverage(part,int(base.sum())),**metric(yy,pp,ll),'equal_asset_mean_nll':float(asset_means.mean())}
                confidence=pp.max(1);correct=pp.argmax(1)==yy;bins=np.minimum((confidence*15).astype(int),14);ece=0.
                for k in range(15):
                    chosen=bins==k;n=int(chosen.sum());acc=float(correct[chosen].mean()) if n else None;conf=float(confidence[chosen].mean()) if n else None
                    if n:ece+=n/len(yy)*abs(acc-conf)
                    calibration.append({'model':model,'seed':seed,'split':split,'population':pop,'bin':k,'lower':k/15,'upper':(k+1)/15,'last_bin_upper_inclusive':k==14,'rows':n,'accuracy':acc,'mean_confidence':conf})
                rec['ece_15_fixed_bins']=ece;metrics.append(rec)
                for asset,nll in asset_means.items():
                    am=mask&ev.asset_id.eq(asset).to_numpy();per_asset_metrics.append({'model':model,'seed':seed,'split':split,'population':pop,'asset_id':asset,'rows':int(am.sum()),'nll':float(nll)})
                for cov in ['origin_hour','current_relative_spread_bin','current_l5_depth_bin']:
                    for group in sorted(ev[cov].unique()):
                        gm=mask&ev[cov].eq(group).to_numpy()
                        if not gm.any():continue
                        subgroups.append({'model':model,'seed':seed,'split':split,'population':pop,'group_kind':cov,'group':group,**coverage(ev[gm],int(mask.sum())),**metric(y[gm],p[gm],loss[gm])})
        log('scored populations/calibration/groups: '+model+' '+str(seed))
    mf=pd.DataFrame(metrics);num=['nll','accuracy','macro_f1','mcc','brier','equal_asset_mean_nll','ece_15_fixed_bins']
    for (model,split,pop),group in mf.groupby(['model','split','population'],sort=False):
        assert sorted(group.seed.tolist())==sorted(SEEDS)
        rec=group.iloc[0].to_dict();rec['seed']='mean_3';rec.update({k:float(group[k].mean()) for k in num});rec['aggregation']='mean of independently trained seed metrics, never averaged probabilities';metrics.append(rec)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False);pd.DataFrame(calibration).to_csv(out/'calibration.csv',index=False);pd.DataFrame(subgroups).to_csv(out/'subgroup_metrics.csv',index=False);pd.DataFrame(per_asset_metrics).to_csv(out/'per_asset_metrics.csv',index=False)
    paired=[];asset_pairs=[];group_pairs=[];daily=[]
    for other in ['UC','UM_V2_DEPTH']:
        for seed in SEEDS+['mean_3']:
            delta=(losses[('UA',seed)]-losses[(other,seed)]) if seed!='mean_3' else np.mean([losses[('UA',s)]-losses[(other,s)] for s in SEEDS],axis=0)
            for split in ['validation','development_holdout']:
                date_order=sorted(ev.loc[ev.split.eq(split),'date'].unique())
                for pop in POPS:
                    mask=ev.split.eq(split).to_numpy()&ev[pop].eq(1).to_numpy();part=ev[mask];d=delta[mask]
                    grouped=pd.DataFrame({'date':part.date.to_numpy(),'difference':d}).groupby('date').difference.agg(['sum','count']).reindex(date_order,fill_value=0)
                    byasset=pd.DataFrame({'asset':part.asset_id.to_numpy(),'difference':d}).groupby('asset').difference.agg(['mean','count'])
                    rec={'namespace':'historical_unmasked_training','model_a':'UA','model_b':other,'seed':seed,'matched_seeds':','.join(map(str,SEEDS)) if seed=='mean_3' else str(seed),'split':split,'population':pop,'rows':len(part),'dates':part.date.nunique(),'assets':part.asset_id.nunique(),'delta_nll':float(d.mean()),'equal_asset_delta_nll':float(byasset['mean'].mean()),'assets_with_negative_delta':int((byasset['mean']<0).sum())}
                    if seed=='mean_3':rec.update(moving_block_interval(grouped['sum'].to_numpy(),grouped['count'].to_numpy(),block_length=5,replicates=10000,seed=20261001))
                    paired.append(rec)
                    for asset,g in byasset.iterrows():asset_pairs.append({'model_a':'UA','model_b':other,'seed':seed,'split':split,'population':pop,'asset_id':asset,'rows':int(g['count']),'delta_nll':float(g['mean'])})
                    for date,g in grouped.iterrows():daily.append({'model_a':'UA','model_b':other,'seed':seed,'split':split,'population':pop,'date':date,'difference_sum':float(g['sum']),'rows':int(g['count'])})
                    if seed=='mean_3':
                        for cov in ['origin_hour','current_relative_spread_bin','current_l5_depth_bin']:
                            for group in sorted(ev[cov].unique()):
                                gm=mask&ev[cov].eq(group).to_numpy()
                                if gm.any():group_pairs.append({'model_a':'UA','model_b':other,'seed':seed,'split':split,'population':pop,'group_kind':cov,'group':group,**coverage(ev[gm],int(mask.sum())),'delta_nll':float(delta[gm].mean())})
    pf=pd.DataFrame(paired);pf.to_csv(out/'paired_differences.csv',index=False);pd.DataFrame(asset_pairs).to_csv(out/'per_asset_differences.csv',index=False);pd.DataFrame(group_pairs).to_csv(out/'subgroup_differences.csv',index=False);pd.DataFrame(daily).to_csv(out/'daily_paired_loss_sums.csv',index=False)
    plots=[]
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(5.5,4.5));cal=pd.DataFrame(calibration)
        for model in MODELS:
            z=cal[(cal.model==model)&(cal.seed==42)&(cal.split=='development_holdout')&(cal.population==POPS[3])&(cal.rows>0)]
            ax.plot(z.mean_confidence,z.accuracy,marker='o',label=model)
        ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='Mean saved confidence',ylabel='Observed accuracy',xlim=(0,1),ylim=(0,1),title='Historical seed 42, P3 development holdout');ax.legend();fig.tight_layout();fig.savefig(out/'calibration_P3_holdout_seed42.svg');plt.close(fig);plots=['calibration_P3_holdout_seed42.svg']
    except ImportError as e:dump(out/'plot_availability.json',{'blocked':str(e),'tables_complete':True})
    manifest={'status':'complete','namespace':'historical_unmasked_training','data_handoff_sha':DATA_SHA,'code_export_sha':h['export_sha'],'input_sha':h['input_sha'],'prediction_index_sha256':sha(docs/'prediction_index.json'),'scoring_source_sha256':sha(Path(__file__)),'runtime':{'python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},'data_checks':checks,'prediction_records':audits,'scoring':{'dtype':'float64','normalize_after_validation':True,'max_allowed_probability_sum_error':1e-5,'clip_min':float(np.finfo(np.float64).eps),'log':'natural','weight':'equal forecast rows','class_order':['Down','Flat','Up'],'multiclass_brier':'sum over all three classes','seed_aggregation':'mean losses, not probability ensemble','anchor_tolerance':1e-7},'bootstrap':{'replicates':10000,'block_length':5,'seed':20261001,'method':'non-circular whole-date blocks; truncate to original date count; pooled counts; percentile95%','conditional_on_historical_selection':True},'train_only_covariates':cutpoints,'training_or_inference_run':False,'no_checkpoint_load':True,'new_runs_namespace_separate':True,'checksums':[]}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name!='replay_manifest.json':manifest['checksums'].append({'path':path.name,'sha256':sha(path),'bytes':path.stat().st_size})
    dump(out/'replay_manifest.json',manifest)
    main_table=pf[(pf.model_b=='UC')&(pf.seed=='mean_3')]
    log(main_table[['split','population','delta_nll','ci_lower','ci_upper','assets_with_negative_delta']].to_string(index=False));log('HISTORICAL_REPLAY_COMPLETE')

if __name__=='__main__':main()
