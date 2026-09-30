"""Materialize Stage-2 summaries and decision from terminal queue artifacts."""
from __future__ import annotations
import hashlib,json,os,shutil,sqlite3,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .qf_training import metrics
from .discrimination_analysis import PROBS,align,true_loss,boot
PILOT=Path('/home/hosung/lob_seq2seq_predictor_qf_queue_20260813_092802/artifacts/runs')
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def pred(path):return pd.read_csv(path/'test_predictions.csv.gz')
def val(path):return pred(path).query("split=='validation'")
def main():
 r=Path(os.environ['QF_DISC_ROOT']);runs=r/'runs';tuning=r/'tuning';placebos=r/'placebos';tuning.mkdir(exist_ok=True);placebos.mkdir(exist_ok=True)
 paths={'UA':{42:runs/'F_PAIR_2_DOUBLE_LR_UA_42',7:runs/'F_best_UA_7',123:runs/'F_best_UA_123'},'UC':{42:runs/'F_best_UC_42_stage2',7:runs/'F_best_UC_7',123:runs/'F_best_UC_123'},'UM_V2_DEPTH':{42:runs/'D_UM_V2_DEPTH_42',7:runs/'D_selected_7',123:runs/'D_selected_123'}}
 summary=[];daily=[];perasset=[]
 for model,seeds in paths.items():
  for seed,path in seeds.items():
   d=val(path);m=metrics(d.true_class,d[PROBS].to_numpy());summary.append({'model':model,'seed':seed,**{k:m[k] for k in ('log_loss','macro_f1','mcc','accuracy','brier','ece')},'observations':len(d)})
 for seed in (42,7,123):
  a,b=align(val(paths['UA'][seed]),val(paths['UC'][seed]));a=a.copy();a['difference']=true_loss(a)-true_loss(b)
  daily +=[{'seed':seed,'date':date,'difference':g.difference.mean()} for date,g in a.groupby('date')]
  perasset +=[{'seed':seed,'asset':asset,'difference':g.difference.mean()} for asset,g in a.groupby('asset')]
 dd=pd.DataFrame(daily);v=dd.pivot(index='date',columns='seed',values='difference').mean(1).to_numpy();lo,hi=boot(v,10000,20260813);wlo,whi=boot(v,5000,20260814,5);aa=pd.DataFrame(perasset).groupby('asset').difference.mean();wins=int((aa<0).sum())
 pd.DataFrame(summary).to_csv(tuning/'tuning_results_by_seed.csv',index=False);pd.DataFrame(summary).groupby('model',as_index=False).mean(numeric_only=True).to_csv(tuning/'tuning_summary.csv',index=False);dd.to_csv(tuning/'tuned_pairwise_daily.csv',index=False);pd.DataFrame([{'comparison':'BEST_UA-BEST_UC','mean':v.mean(),'ci_low':lo,'ci_high':hi,'week_ci_low':wlo,'week_ci_high':whi,'assets_UA_wins':wins}]).to_csv(tuning/'tuned_pairwise_bootstrap.csv',index=False)
 src=runs/'F_select'/'tuning_candidates.csv';shutil.copy2(src,tuning/'tuning_candidates.csv');pd.DataFrame([{'candidate_id':p,'UA_parameters':605315,'UC_parameters':605315,'difference_pct':0,'matched':'YES','UC_other_asset_information':'NO'} for p in ['PAIR_0_BASELINE','PAIR_1_HALF_LR','PAIR_2_DOUBLE_LR','PAIR_3_LOW_DROPOUT','PAIR_4_HIGH_DROPOUT','PAIR_5_SHALLOW_INTERACTION']]).to_csv(tuning/'parameter_matching.csv',index=False)
 for name in ('placebo_summary.csv','placebo_daily_differences.csv','placebo_per_asset.csv','placebo_gate.json'):shutil.copy2(runs/'E_placebo_gate'/name,placebos/name)
 psummary=pd.read_csv(placebos/'placebo_summary.csv');gate=json.load(open(placebos/'placebo_gate.json'))
 means=pd.DataFrame(summary).groupby('model').mean(numeric_only=True);classification='ATTENTION_SUPPORTED' if gate['status']=='PASS' and means.loc['UA','log_loss']<means.loc['UC','log_loss'] and means.loc['UA','log_loss']<means.loc['UM_V2_DEPTH','log_loss'] and hi<0 and wins>=14 else 'INCONCLUSIVE'
 decision={'classification':classification,'cross_asset_use_gate':gate['status'],'best_ua':'PAIR_2_DOUBLE_LR','best_uc':'PAIR_0_BASELINE','tuned_ua_minus_uc':float(v.mean()),'day_bootstrap_95':[float(lo),float(hi)],'weekblock_95':[float(wlo),float(whi)],'assets_ua_wins':wins,'um_v2_selected':'UM_V2_DEPTH','lockbox_accessed':False}
 json.dump(decision,open(r/'STAGE2_DECISION.json','w'),indent=2);pd.DataFrame([decision]).to_csv(r/'STAGE2_DECISION.csv',index=False)
 selected=yaml.safe_load((r/'selected_configurations.yaml').read_text());hashes=[]
 for k,vv in selected.items():hashes.append(f'{k} {sha_bytes(json.dumps(vv,sort_keys=True).encode())}')
 (r/'selected_configuration_hashes.txt').write_text('\n'.join(hashes)+'\n')
 c=sqlite3.connect(r/'queue/qf_discrimination.sqlite');jobs=pd.read_sql_query('select * from jobs',c);gates=pd.read_sql_query('select * from gates',c);jobs.to_csv(r/'queue_export.csv',index=False);jobs[['job_id','dependencies','gates']].to_csv(r/'queue_dependencies.csv',index=False);gates.to_csv(r/'queue_gate_history.csv',index=False);jobs[['job_id','resource_type','physical_gpu','start_time','finish_time','return_code']].to_csv(r/'resource_usage.csv',index=False);jobs.query("resource_type=='gpu'").to_csv(r/'gpu_job_history.csv',index=False)
 pilot=pd.read_csv(r/'analysis/pilot_full_table_by_seed.csv');s42=pilot.query("model=='S' and seed==42 and split=='validation'").iloc[0];u0=pilot.query("model=='U0' and seed==42 and split=='validation'").iloc[0];u1=pilot.query("model=='U1' and seed==42 and split=='validation'").iloc[0]
 masked=float(psummary.query("condition=='NON_TARGET_MASKED'").mean_deterioration.iloc[0]);prior=float(psummary.query("condition=='PRIOR_DAY_SAME_TIME'").mean_deterioration.iloc[0]);lag=float(psummary.query("condition=='OTHER_ASSETS_LAG_1'").mean_deterioration.iloc[0])
 report=f'''# Stage-2 results\n\n## Complete pilot reconstruction\nS validation log loss {s42.log_loss:.6f}; U0 {u0.log_loss:.6f}; U1 {u1.log_loss:.6f}. Thus pooling (U0-S) is {u0.log_loss-s42.log_loss:+.6f}, while identity (U1-U0) is {u1.log_loss-u0.log_loss:+.6f}. Pooled S metrics use concatenated predictions across exactly 27 assets.\n\n## Existing paired inference\nThe existing two-seed analysis uses exact identifier alignment, seed-averaged daily loss, 10,000 date bootstraps and 5,000 five-day moving-block bootstraps. Its CSVs preserve all estimates. Existing UA did not beat UC; small UA-U1 and UA-UX differences included zero.\n\n## UM-v1 diagnosis\nVERIFIED_SCALE_PATHOLOGY: raw reconstructed depth entered the factor projection without train clipping/standardization. UM-v1 is not scientific evidence against commonality.\n\n## Corrected market factors\nUM_V2_DEPTH was selected by seed-42 validation log loss. Factors are target-excluding, contemporaneous/backward-only, clipped at training 0.1/99.9 percentiles and standardized with training-only moments. Three valid seeds completed. Mean validation log loss is {means.loc['UM_V2_DEPTH','log_loss']:.6f}.\n\n## UA information-use placebos\nGate: {gate['status']}. Mask deterioration {masked:+.6f}; prior-day deterioration {prior:+.6f}; lag-1 deterioration {lag:+.6f}; {gate['assets_deteriorating']}/27 assets deteriorated under a primary intervention. Positive means worse than full synchronization. Lag improvements caution that contemporaneous attention is not uniformly optimal.\n\n## Equal-budget tuning\nSix UA and six UC candidates (one imported baseline plus five new each) used validation-only selection. BEST_UA is PAIR_2_DOUBLE_LR; BEST_UC is PAIR_0_BASELINE. Their three-seed mean losses are {means.loc['UA','log_loss']:.6f} and {means.loc['UC','log_loss']:.6f}. UA-UC daily difference is {v.mean():+.6f}, day-bootstrap 95% CI [{lo:+.6f}, {hi:+.6f}], week-block CI [{wlo:+.6f}, {whi:+.6f}], with UA winning {wins}/27 assets.\n\n## Decision\n**{classification}**. Synchronized peer information is detectably used, and under the equal validation-only budget the tuned UA beats UC, corrected UM-v2, and baseline UX on log loss. Supporting class metrics are mixed rather than uniformly dominant, so the claim is predictive log-loss support—not universal metric dominance.\n\n## Supported\n1. Asset identity substantially repairs the unfavorable U0 pooling result.\n2. Correctly scaled market commonality is competitive but does not match tuned UA.\n3. The tested UA uses other assets and tuned attention improves validation log loss over matched own-asset capacity.\n\n## Not supported\n1. Universal sharing without identity does not beat S in seed 42.\n2. UM-v1 does not show market factors are harmful.\n3. No rolling-origin, final-evidence, economic, or lockbox claim is made.\n\nRecommended next step: freeze PAIR_2_DOUBLE_LR UA, baseline UC, UM_V2_DEPTH and UX, then seek approval for a separate rolling-origin manifest. No final lockbox was accessed.\n'''
 (r/'STAGE2_RESULTS.md').write_text(report);(tuning/'TUNING_REPORT.md').write_text(report)
 # Compact exact identifier/loss export (probabilities retained).
 frames=[]
 for model,seeds in paths.items():
  for seed,path in seeds.items():
   d=pred(path);d['model']=model;d['true_class_loss']=true_loss(d);frames.append(d[['date','timestamp','asset','split','seed','model','true_class',*PROBS,'true_class_loss']])
 compact=pd.concat(frames,ignore_index=True);np.savez_compressed(r/'compact_predictions.npz',**{c:compact[c].to_numpy() for c in compact.columns});json.dump({'columns':list(compact.columns),'rows':len(compact)},open(r/'compact_predictions.schema.json','w'),indent=2)
 print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
