"""Deterministic finite global DAG; no implicit model/seed expansion."""
import argparse,csv,json
import pandas as pd
from common import *

def generate():
    raw=pd.read_csv(ROOT/'docs/qf_data_review/20261001/split_dates.csv',dtype=str)
    dates=pd.to_datetime(raw.date,format='%Y%m%d').dt.strftime('%Y-%m-%d').tolist()
    assets=sorted(pd.read_csv(ROOT/'docs/qf_data_review/20261001/workbook_cache_comparison.csv').asset_id.tolist())
    assert len(dates)==493 and len(assets)==27
    splits={}
    for fold,(a,tr,va,end) in SPLIT_INDICES.items():
        splits[fold]={'indices_zero_based_half_open':[a,tr,va,end],
                      'train':dates[a:tr],'validation':dates[tr:va],
                      ('development_holdout' if fold=='main' else 'rolling_evaluation'):dates[va:end]}
    protocol={'id':GROUP,'version':'p3-v1','data_handoff_sha':HANDOFF,'input_lineage':'073cd89b556224eeae22413c3c05188a947e990a',
              'input_manifest_sha256':sha(ROOT/'recovery/input_manifest.csv'),
              'label_contract_sha256':sha(ROOT/'docs/qf_data_review/20261001/data_contract.json'),
              'forecast_rows_sha256':sha(ROOT/'artifacts/qf_data_review/20261001/forecast_rows.csv.gz'),
              'row_key_label_digest':read(ROOT/'docs/qf_data_review/20261001/data_checks.json')['row_key_label_sha256'],
              'assets':assets,'dates':dates,'splits':splits,'split_manifest_hash':digest(splits),
              'features':'13 recovered no-investor features; full same-day history from09:00; no off-grid seeding; forward fill then zero fill',
              'labels':'float32 bake midpoint and return; NumPy1.22 scalar comparison to Python +/-0.0001; Down0 Flat1 Up2; exact frozen labels required',
              'loss_mask':'P3 only, applied only to loss/selection; input panel/history retained; no future-dependent peer availability',
              'peer_policy':'unchanged legacy filled full panel, self-inclusive; no new availability mask',
              'training':BASE,'lrs':LRS,'selection_seed':42,'confirmation_seeds':[7,123],
              'checkpoint_rule':'validation P3 row-weighted natural-log NLL improves accepted best by >0.0001; min20 patience30 max200',
              'lr_selection':'lowest selected-checkpoint validation P3 NLL; ties within1e-8 choose smaller LR',
              'accumulation':'sum valid-row CE gradients; divide by total valid-row count in each32-day window before clipping/step; final partial window same rule',
              'scaler_fit':'all train-date origin positions/asset factors, never validation/eval; recovered six-factor LOO train .001/.999 clipping+mean/std',
              'analysis_covariates':'current original pre-fill spread and L5 depth; train-only pooled tertiles; invalid separate; fixed hourly origin buckets',
              'models':{'U1':'recovered U1','UC':'recovered third own causal layer;605315 parameters',
                        'UA':'recovered third same-time cross layer;605315; self included','UM_V2_DEPTH':'recovered StandardizedMarketModel six factors and discriminator scaler',
                        'UX':'recovered static learned NxN residual including self','GRU_U1':'input Linear13→128, asset embedding, causal2-layer GRU128/dropout.1 between layers, no position embedding/input dropout, Linear128→3',
                        'SHARED_QUERY':'UA identical parameters; query uses WQ(mean current states), KV unchanged, own residual unchanged',
                        'U0':'recovered U0; no explicit asset embedding; indirect identity not removed','S':'separate recovered own-only parameter set per asset/seed',
                        'TLOB_ADAPTED':'optional official pinned TLOB, adapter manifest required; causal per-origin prefixes; no prior TLOB execution before adapter gates'},
              'fixed_decomposition':{'lr':.0001,'seeds':SEEDS,'S_assets':27,'reuse_exact_matching_U0_U1':True},
              'rolling':'R1 exact main reuse with evaluation subset; R2/R3 refit own train transforms; main-selected LR; retrospective',
              'metrics':{'probabilities':'float64 normalize after finite,[0,1],positive-sum checks','nll':'natural log; clip true-class probability at float64 epsilon; equal row weights',
                         'macro_f1':'fixed3 classes','brier':'sum over3 classes','equal_asset_nll':True,'seed_summary':'average losses not probabilities',
                         'calibration_bins':15,'bootstrap':{'draws':10000,'block_dates':5,'seed':20261001,'resample_unit':'whole date panels; assets/models/seeds stay together; fold boundaries preserved','interval':'95% percentile'}},
              'interventions':['UA self-only cross attention','UA target-current query/own KV, other KV at t-1; original full history/positions; common P3 support; no wrap'],
              'owners':OWNERS,'historical_namespace_separate':True,'holdout_is_untouched':False,
              'limits':['unknown provider aggregation/fill/release semantics','historically inspected retrospective data','three optimization seeds','no executable trading profitability claim'],
              'revisions':'Core semantics immutable per protocol version. Optional TLOB adapter may add files/digest while frozen core is byte-identical. Scientific corrections require versioned invalidation.',
              'retries':{'transient':2,'oom_rebatch_once':True,'effective_days_unchanged':32}}
    write(DOCS/'protocol.json',protocol);ph=digest(protocol)
    write(CONFIG/'splits.json',splits)
    records=[]
    for fold,groups in splits.items():
        for split,values in groups.items():
            if split!='indices_zero_based_half_open':records.extend(dict(fold=fold,split=split,date=d) for d in values)
    pd.DataFrame(records).to_csv(CONFIG/'split_dates.csv',index=False)
    jobs=[]
    def add(id,owner,kind,priority,resource='cpu',deps=None,**cfg):
        record=dict(job_id=id,owner=owner,kind=kind,priority=priority,resource=resource,deps=deps or [],config=cfg,protocol_hash=ph)
        record['job_spec_hash']=digest(record);jobs.append(record);return id
    for owner in ['a5000','rtx3090']:add('gate_'+owner,owner,'gate',0)
    add('adapter_TLOB_ADAPTED','a5000','adapter',40,deps=['gate_a5000'],model='TLOB_ADAPTED')
    for fold in SPLIT_INDICES:add('prior_'+fold,'a5000','prior',1,deps=['gate_a5000'],fold=fold)
    add('historical_replay','rtx3090','historical',1,deps=['gate_rtx3090'])
    selected={};fixed={}
    for model in FAMILIES:
        owner=OWNERS[model];p=10 if model in FAMILIES[:6] else 20 if model=='SHARED_QUERY' else 40 if model=='TLOB_ADAPTED' else 50
        gate=['gate_'+owner]+(['adapter_TLOB_ADAPTED'] if model=='TLOB_ADAPTED' else [])
        trials=[add(fit_name(model,42,lr),owner,'fit',p,'gpu',gate,model=model,seed=42,lr=lr,fold='main',family='lr_search') for lr in LRS]
        selector=add('select_'+model,owner,'select',p,deps=trials,model=model,trials=trials)
        selected[model]={42:add('selected_'+model+'_42',owner,'alias',p,deps=[selector],model=model,seed=42,select_from=selector,fold='main',family='selected')}
        for seed in [7,123]:
            selected[model][seed]=add(fit_name(model,seed),owner,'fit',p,'gpu',[selector],model=model,seed=seed,select_from=selector,fold='main',family='selected')
        for seed in SEEDS:add('analyze_'+model+'_'+str(seed),owner,'analysis',p+1,deps=[selected[model][seed]],source_job=selected[model][seed],model=model,seed=seed)
        if model in ['U0','U1']:
            fixed[model]={42:add('fixed_'+model+'_42',owner,'alias',50,deps=[fit_name(model,42,.0001)],source_job=fit_name(model,42,.0001),model=model,seed=42,family='fixed')}
            for seed in [7,123]:
                fixed[model][seed]=add('fixed_'+model+'_'+str(seed),owner,'fixed_or_alias',50,'gpu',[selected[model][seed]],model=model,seed=seed,lr=.0001,fold='main',source_job=selected[model][seed],family='fixed')
        if model in ['UA','UC','UM_V2_DEPTH']:
            for seed in SEEDS:
                add('R1_'+model+'_'+str(seed),owner,'rolling_reuse',30,deps=[selected[model][seed]],source_job=selected[model][seed],model=model,seed=seed,fold='R1')
                for fold in ['R2','R3']:
                    job=add(fit_name(model,seed,fold=fold),owner,'fit',30,'gpu',[selector],model=model,seed=seed,select_from=selector,fold=fold,family='rolling')
                    add('analyze_'+job,owner,'analysis',31,deps=[job],source_job=job,model=model,seed=seed)
        if model=='UA':
            for seed in SEEDS:add('interventions_UA_'+str(seed),owner,'intervention',35,'gpu',[selected[model][seed]],source_job=selected[model][seed],model=model,seed=seed,fold='main')
    for seed in SEEDS:
        fits=[add(fit_name('S',seed,.0001,asset=asset),'a5000','fit',50,'gpu',['gate_a5000'],model='S',seed=seed,lr=.0001,fold='main',asset=asset,family='fixed') for asset in assets]
        add('pool_S_'+str(seed),'a5000','pool',51,deps=fits+list(fixed[m][seed] for m in ['U0','U1']),model='S',seed=seed,source_jobs=fits,fixed_jobs={m:fixed[m][seed] for m in ['U0','U1']})
    for owner in ['a5000','rtx3090']:
        local=[j['job_id'] for j in jobs if j['owner']==owner]
        add('final_'+owner,owner,'final',60,deps=local,allow_terminal_dependencies=True)
    counts={owner:{'jobs':sum(j['owner']==owner for j in jobs),'unconditional_fits':sum(j['owner']==owner and j['kind']=='fit' for j in jobs),
                   'conditional_fits_or_aliases':sum(j['owner']==owner and j['kind']=='fixed_or_alias' for j in jobs)} for owner in OWNERS.values()}
    manifest={'run_group':GROUP,'protocol_hash':ph,'jobs':jobs,'counts_by_owner':counts,
              'full_training_fit_bounds':[153,157],'main_shared_fits':54,'S_fits':81,'rolling_new_fits':18,
              'conditional_additional_fixed_fits':4,'R1_reuses':9,'seed42_selected_reuses':9,'fixed_seed42_reuses':2}
    write(CONFIG/'queue_manifest.json',manifest)
    flat=[{**{k:j[k] for k in ['job_id','owner','kind','priority','resource','job_spec_hash']},'dependencies':json.dumps(j['deps']),'config':json.dumps(j['config'],sort_keys=True)} for j in jobs]
    pd.DataFrame(flat).to_csv(CONFIG/'queue_manifest.csv',index=False)
    print(json.dumps({'protocol_hash':ph,'counts':counts,'total_jobs':len(jobs)}))

if __name__=='__main__':generate()
