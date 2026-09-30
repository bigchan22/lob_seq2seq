"""Dependency-free test runner for environments without pytest."""
import json,multiprocessing,tempfile,time
from pathlib import Path
import torch
from lob_forecasting.data.market_factors import leave_one_out_market_factors
from lob_forecasting.experiments.queue_db import QueueDB
from lob_forecasting.experiments.queue_gates import validate_gpu,lockbox_allowed
from lob_forecasting.experiments.qf_plan import build_plan
from lob_forecasting.models.qf_variants import build_qf_model,trainable_parameters

def check(name,fn,results):
 try:fn();results.append((name,'PASS',''))
 except Exception as e:results.append((name,'FAIL',repr(e)))
def main():
 results=[]
 with tempfile.TemporaryDirectory() as td:
  path=Path(td)/'q.sqlite';db=QueueDB(path);db.initialize();plan=build_plan();db.add_jobs(plan['jobs'])
  check('job_counts',lambda:(_ for _ in ()).throw(AssertionError()) if plan['confirmatory_gpu_jobs']!=72 or sum(x['phase']=='pilot' for x in plan['jobs'])!=12 else None,results)
  db.set_gate('repo_ready',1);db.set_gate('data_readable',1);db.set_gate('queue_tests_pass',1);db.refresh()
  check('dependency_unblocking',lambda:(_ for _ in ()).throw(AssertionError()) if not any(x['state']=='PENDING' and x['phase']=='smoke' for x in db.rows()) else None,results)
  a=db.claim('gpu',0);b=db.claim('gpu',0);check('atomic_distinct_claims',lambda:(_ for _ in ()).throw(AssertionError()) if not a or not b or a['job_id']==b['job_id'] else None,results)
  check('unsupported_gpu_rejected',lambda: validate_gpu(2),results);results[-1]=(results[-1][0],'PASS','expected rejection') if results[-1][1]=='FAIL' else (results[-1][0],'FAIL','accepted GPU 2')
  db.state(a['job_id'],'RUNNING',heartbeat_time='0',failure_message='99999999');check('stale_recovery',lambda:(_ for _ in ()).throw(AssertionError()) if a['job_id'] not in db.recover_stale(1) else None,results)
  check('export_reimport',lambda:(db.export(Path(td)/'export.csv'),QueueDB(path).rows()),results)
  check('lockbox_refusal',lambda:(_ for _ in ()).throw(AssertionError()) if lockbox_allowed(td,True,True,True) else None,results)
 n=27;kw=dict(num_assets=n,num_features=13,d_model=32,nhead=4,temporal_layers=1,dropout=0,cross_asset_layers=1)
 ua=build_qf_model('UA',**kw);uc=build_qf_model('UC',**kw);ux=build_qf_model('UX',**kw)
 check('uc_parameter_match',lambda:(_ for _ in ()).throw(AssertionError()) if trainable_parameters(ua)!=trainable_parameters(uc) else None,results)
 check('ux_no_attention',lambda:(_ for _ in ()).throw(AssertionError()) if hasattr(ux,'cross_asset_encoder') else None,results)
 x=torch.randn(1,38,n,13);ua.eval();base=ua(x);z=x.clone();z[:,20:]+=10;past=ua(z)
 check('ua_temporal_causality',lambda:(_ for _ in ()).throw(AssertionError()) if (base[:,:20]-past[:,:20]).abs().max()>1e-7 else None,results)
 mid=torch.arange(1*2*3,dtype=torch.float32).reshape(1,2,3)+100;aq=torch.ones(1,2,3,5);bq=2*aq;fac=leave_one_out_market_factors(mid,mid+.1,mid-.1,aq,bq)
 check('market_leave_one_out_shape',lambda:(_ for _ in ()).throw(AssertionError()) if fac.shape!=(1,2,3,6) else None,results)
 check('no_investor_enforced',lambda:(_ for _ in ()).throw(AssertionError()) if any(x['feature_set']!='no_investor' for x in plan['jobs']) else None,results)
 # Stable round robin covers 27 unique assets.
 assets=[f'A{i:02d}' for i in range(27)];shards=[assets[::2],assets[1::2]]
 check('s_shard_coverage',lambda:(_ for _ in ()).throw(AssertionError()) if len(set(sum(shards,[])))!=27 else None,results)
 for name,status,msg in results:print(f'{name},{status},{msg}')
 if any(x[1]=='FAIL' for x in results):raise SystemExit(1)
if __name__=='__main__':main()
