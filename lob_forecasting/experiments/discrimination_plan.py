"""Dependency graph for Stage-2 discrimination; no rolling or lockbox jobs exist here."""
from __future__ import annotations
import os
PY='/home/hosung/miniconda3/envs/lob/bin/python3.10'
PHASES=('A_IMPORT_AND_RECONSTRUCT','B_EXISTING_ANALYSIS','C_UM_V1_DIAGNOSIS','D_UM_V2','E_UA_PLACEBOS','F_UA_UC_TUNING','G_CORRECTED_COMPARISON','H_POOLING_PIVOT','I_FINAL_DECISION','J_PACKAGE')
def build_plan():
 jobs=[]
 def add(jid,phase,resource='cpu',deps=(),gates=(),model='',seed=42,priority=500,config=''):
  jobs.append(dict(job_id=jid,experiment_id=jid,phase=phase,model=model,asset_shard='',seed=seed,fold='development',horizon=1,representation='state',protocol='qf_discrimination_dev_v1',feature_set='no_investor',resource_type=resource,physical_gpu='',priority=priority,dependencies=list(deps),gates=list(gates),command=f'{PY} -m lob_forecasting.experiments.discrimination_runner --job-id {jid}',run_directory=f'runs/{jid}',state='PENDING' if not deps and not gates else 'BLOCKED',retry_count=0,max_retries=1 if resource=='gpu' else 0,resolved_config_hash=config,input_run_hashes=''))
 add('A_import','A_IMPORT_AND_RECONSTRUCT');add('A_reconstruct','A_IMPORT_AND_RECONSTRUCT',deps=('A_import',));add('B_existing_analysis','B_EXISTING_ANALYSIS',deps=('A_reconstruct',));add('C_um_v1_diagnosis','C_UM_V1_DIAGNOSIS',deps=('A_import',));add('D_factor_tests','D_UM_V2',deps=('A_import',));add('E_placebo_tests','E_UA_PLACEBOS',deps=('A_import',))
 add('D_smoke_LITE','D_UM_V2',deps=('D_factor_tests',),model='UM_V2_LITE');add('D_smoke_DEPTH','D_UM_V2',deps=('D_factor_tests',),model='UM_V2_DEPTH')
 add('E_placebo_42','E_UA_PLACEBOS','gpu',('E_placebo_tests',),model='UA',seed=42,priority=900);add('E_placebo_7','E_UA_PLACEBOS','gpu',('E_placebo_tests',),model='UA',seed=7,priority=899);add('E_placebo_gate','E_UA_PLACEBOS',deps=('E_placebo_42','E_placebo_7'))
 add('D_UM_V2_LITE_42','D_UM_V2','gpu',('D_smoke_LITE',),model='UM_V2_LITE',priority=950);add('D_UM_V2_DEPTH_42','D_UM_V2','gpu',('D_smoke_DEPTH',),model='UM_V2_DEPTH',priority=949);add('D_select','D_UM_V2',deps=('D_UM_V2_LITE_42','D_UM_V2_DEPTH_42'))
 add('D_selected_7','D_UM_V2','gpu',('D_select',),('um_v2_valid',),model='UM_V2_SELECTED',seed=7,priority=880);add('D_selected_123','D_UM_V2','gpu',('D_selected_7',),('um_v2_seed123_allowed',),model='UM_V2_SELECTED',seed=123,priority=870)
 pairs=['PAIR_1_HALF_LR','PAIR_2_DOUBLE_LR','PAIR_3_LOW_DROPOUT','PAIR_4_HIGH_DROPOUT','PAIR_5_SHALLOW_INTERACTION']
 for i,pair in enumerate(pairs):
  for model in ('UA','UC'):add(f'F_{pair}_{model}_42','F_UA_UC_TUNING','gpu',('E_placebo_gate',),('cross_asset_gate_nonfail',),model=model,seed=42,priority=800-i,config=pair)
 tune=[f'F_{p}_{m}_42' for p in pairs for m in ('UA','UC')]
 add('F_select','F_UA_UC_TUNING',deps=tuple(tune),gates=('cross_asset_gate_nonfail',));add('F_best_UA_7','F_UA_UC_TUNING','gpu',('F_select',),model='BEST_UA',seed=7);add('F_best_UC_7','F_UA_UC_TUNING','gpu',('F_select',),model='BEST_UC',seed=7);add('F_best_UA_123','F_UA_UC_TUNING','gpu',('F_best_UA_7',),model='BEST_UA',seed=123);add('F_best_UC_123','F_UA_UC_TUNING','gpu',('F_best_UC_7',),model='BEST_UC',seed=123)
 add('F_compare','F_UA_UC_TUNING',deps=('F_best_UA_123','F_best_UC_123'));add('G_corrected_compare','G_CORRECTED_COMPARISON',deps=('D_selected_123','F_compare'));add('H_pooling_existing','H_POOLING_PIVOT',deps=('B_existing_analysis','F_compare'));add('I_final_decision','I_FINAL_DECISION',deps=('G_corrected_compare','H_pooling_existing','E_placebo_gate'));add('J_package','J_PACKAGE',deps=('I_final_decision',))
 return {'jobs':jobs}
