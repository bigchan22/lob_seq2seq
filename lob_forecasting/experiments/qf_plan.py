"""Deterministic staged job plan. Confirmatory count is asserted at construction."""
import hashlib,json
MODELS=['U0','U1','UM','UX','UC','UA'];BASE={'horizon':'1','representation':'state','feature_set':'no_investor'}
def job(job_id,phase,model='',resource_type='gpu',priority=0,deps=(),gates=(),seed=42,fold='development',protocol='qf_decomposition_dev_v1',shard=''):
 d=dict(job_id=job_id,experiment_id=job_id,phase=phase,model=model,asset_shard=shard,seed=str(seed),fold=fold,protocol=protocol,resource_type=resource_type,physical_gpu='',priority=str(priority),dependencies=list(deps),gates=list(gates),command=f"/home/hosung/miniconda3/envs/lob/bin/python3.10 -m lob_forecasting.experiments.qf_runner --job-id {job_id}",run_directory=f"artifacts/runs/{job_id}",git_commit='',resolved_config_hash='',data_manifest_hash='',state='BLOCKED',**BASE);return d
def build_plan():
 jobs=[]
 pre=['repo_ready','data_readable','gpu_0_ready','gpu_1_ready','queue_tests_pass','model_smoke_pass','causality_pass','artifact_writer_pass']
 for g in pre:jobs.append(job('preflight_'+g,'preflight',resource_type='cpu',priority=1000,gates=(),protocol='qf_decomposition_dev_v1'))
 smoke=['S','U0','U1','UM','UX','UC','UA']
 for i,m in enumerate(smoke):jobs.append(job('smoke_'+m,'smoke',m,priority=950-i,gates=('repo_ready','data_readable','queue_tests_pass')))
 pilot=[('pilot_UA','UA',900,''),('pilot_UM','UM',890,''),('pilot_UC','UC',880,''),('pilot_UX','UX',870,''),('pilot_U1','U1',860,''),('pilot_U0','U0',850,''),('pilot_S_shard_A','S',840,'A'),('pilot_S_shard_B','S',840,'B')]
 for jid,m,p,s in pilot:jobs.append(job(jid,'pilot',m,resource_type='gpu',priority=p,gates=('model_smoke_pass','causality_pass','artifact_writer_pass'),shard=s))
 jobs += [job('pilot_S_aggregate','pilot','S',resource_type='cpu',priority=820,deps=('pilot_S_shard_A','pilot_S_shard_B')),job('pilot_compare','pilot',resource_type='cpu',priority=810,deps=tuple(x[0] for x in pilot)+('pilot_S_aggregate',)),job('pilot_gate','pilot',resource_type='cpu',priority=800,deps=('pilot_compare',)),job('pilot_package','pilot',resource_type='cpu',priority=790,deps=('pilot_gate',))]
 confirm=[]
 for seed in [42,7,123]:
  for fold in ['fold_1','fold_2','fold_3']:
   for m in ['S_A','S_B',*MODELS]:confirm.append(job(f"rolling_{m}_{seed}_{fold}",'confirmatory','S' if m.startswith('S_') else m,priority=500,seed=seed,fold=fold,protocol='qf_rolling_v1',shard=m[-1] if m.startswith('S_') else '',gates=('pilot_attention_gate_pass','protocol_qf_rolling_ready','rolling_fold_manifest_approved')))
 assert len(confirm)==72;jobs+=confirm
 for seed in [42,7,123]:
  for fold in ['fold_1','fold_2','fold_3']:
   deps=tuple(f"rolling_{m}_{seed}_{fold}" for m in ['S_A','S_B',*MODELS])
   agg=f'rolling_aggregate_{seed}_{fold}';jobs.append(job(agg,'confirmatory',resource_type='cpu',priority=480,seed=seed,fold=fold,protocol='qf_rolling_v1',deps=deps,gates=('pilot_attention_gate_pass','protocol_qf_rolling_ready','rolling_fold_manifest_approved')));jobs.append(job(f'rolling_compare_{seed}_{fold}','confirmatory',resource_type='cpu',priority=470,seed=seed,fold=fold,protocol='qf_rolling_v1',deps=(agg,),gates=('pilot_attention_gate_pass','protocol_qf_rolling_ready','rolling_fold_manifest_approved')))
 for m in ['U1','UM','UX','UC','UA']:jobs.append(job(f'pilot_seed7_{m}','pilot_seed7',m,priority=700,seed=7,gates=('pilot_attention_gate_marginal',)))
 jobs.append(job('pilot_seed7_compare','pilot_seed7',resource_type='cpu',priority=690,seed=7,deps=tuple(f'pilot_seed7_{m}' for m in ['U1','UM','UX','UC','UA']),gates=('pilot_attention_gate_marginal',)))
 for x in ['full_synchronized','non_target_masked','other_assets_lag_1','other_assets_lag_2','other_assets_lag_3','other_day_same_time','asset_identity_permutation','market_factor_only']:jobs.append(job('placebo_'+x,'placebo','UA',resource_type='cpu',priority=400,deps=('pilot_UA',),gates=('pilot_attention_gate_pass_or_marginal',)))
 for m in ['U1','UM','UA']:
  for h in [1,2,3,6]:
   for rep in ['state','change','state_plus_change']:
    z=job(f'explore_{m}_h{h}_{rep}','exploration',m,priority=300,gates=('core_pilot_complete',));z['horizon']=str(h);z['representation']=rep;jobs.append(z)
 for m in ['EL_own','EL_cross','GBDT_own','GBDT_cross']:jobs.append(job('control_'+m,'controls',m,'cpu',200,gates=('core_pilot_complete',)))
 for x in ['statistical_inference','economic_diagnostics']:jobs.append(job('analysis_'+x,'analysis',resource_type='cpu',priority=100,gates=('confirmatory_predictions_ready',)))
 return {'schema_version':1,'jobs':jobs,'confirmatory_gpu_jobs':len(confirm)}
