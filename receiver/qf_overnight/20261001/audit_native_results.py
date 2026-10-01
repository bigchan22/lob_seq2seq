"""Independent support/hash checks and matched analysis of native saved outputs.
No checkpoint loading, model imports or inference. Never writes shared source.
"""
from __future__ import annotations
import argparse,gzip,hashlib,io,json,os,subprocess
from pathlib import Path
import numpy as np
import pandas as pd

DATA_SHA='e85a89fd56584b513be940e6259c637369bd830b'
KEYS=['asset_id','date','origin_time','split','fold']
POPS=['P0_legacy','P1_valid_cache_endpoints','P2_candidate_clock','P3_intersection']
PROBS=['probability_down','probability_flat','probability_up']
SEEDS=[42,7,123]
MODELS=['UA','UC','UM_V2_DEPTH','UX','U1','GRU_U1','SHARED_QUERY','U0','TLOB_ADAPTED']

def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha(x):return hashlib.sha256(x).hexdigest()
def write_table(p,rows,fields):
 tmp=p.with_suffix('.tmp');pd.DataFrame(rows,columns=None if rows else fields).to_csv(tmp,index=False);tmp.replace(p)
class Audit:
 def __init__(self,c):
  self.c=c;self.bridge=Path(c['bridge']);self.root=Path(c['runtime_root']);self.cache=self.root/'audit_blobs';self.cache.mkdir(exist_ok=True)
  self.source=Path(c['native_source']);self.ready=json.loads((self.source/'coordination/qf_overnight/20261001/ready.json').read_text())
  self.env=dict(os.environ,GIT_LFS_SKIP_SMUDGE='1',GIT_TERMINAL_PROMPT='0',GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15')
  self.refs={};self.indices={};self.missing=[];self.frames={};self.provenance=[]
  sys=__import__('sys');sys.path.insert(0,str(self.source/'scripts/qf_overnight'))
  from scoring import metrics,paired_bootstrap,losses
  self.metric=metrics;self.bootstrap=paired_bootstrap;self.losses=losses
  self.rows=pd.read_csv(self.source/'artifacts/qf_data_review/20261001/forecast_rows.csv.gz',dtype={'date':str,'asset_id':str,'origin_time':str,'endpoint_time':str})
  self.dates=sorted(self.rows.date.unique());self.bounds={'main':[0,345,394,493],'R1':[0,345,394,427],'R2':[0,378,427,460],'R3':[0,411,460,493]}
 def git(self,*a):return subprocess.run(['git','-C',str(self.bridge),*a],env=self.env,capture_output=True,check=True,timeout=120).stdout
 def blob(self,commit,path):
  oid=self.git('rev-parse',commit+':'+path).decode().strip();target=self.cache/oid
  if not target.exists():
   raw=self.git('cat-file','blob',oid);temp=target.with_suffix('.tmp');temp.write_bytes(raw);temp.replace(target)
  return target.read_bytes()
 def receive(self,owner):
  ref='refs/heads/results/qf-overnight-20261001-'+owner
  lines=self.git('ls-remote','--heads','origin',ref).decode().splitlines()
  if not lines:self.missing.append({'owner':owner,'reason':'result branch unavailable'});return
  commit=lines[0].split()[0];self.refs[owner]=commit
  # A private tracking ref prevents racing the common publisher's ref update.
  self.git('fetch','--filter=blob:none','--depth=4','--no-tags','--no-recurse-submodules','origin',ref+':refs/remotes/origin/receiver-audit-'+owner)
  prefix='results/qf_overnight/20261001/'+owner+'/'
  try:index=json.loads(self.blob(commit,prefix+'export_manifest.json'))
  except subprocess.CalledProcessError:self.missing.append({'owner':owner,'reason':'native export_manifest not published yet'});return
  assert index['owner']==owner and index['protocol_hash']==self.ready['protocol_hash']
  self.indices[owner]=(commit,prefix,{r['job_id']:r for r in index['runs']})
 def expected(self,fold):
  a,t,v,e=self.bounds[fold];f=self.rows[self.rows.date.isin(self.dates[t:e])].copy();f['fold']=fold
  f['split']=np.where(f.date.isin(self.dates[t:v]),'validation','development_holdout' if fold=='main' else 'rolling_evaluation')
  return f.sort_values(KEYS).reset_index(drop=True)
 def resolve(self,owner,id,trail=()):
  if owner not in self.indices:return None
  commit,prefix,records=self.indices[owner]
  if id not in records:return None
  assert id not in trail,'alias cycle';record=records[id];result=record['result']
  if result.get('reused_from'):return self.resolve(owner,result['reused_from'],trail+(id,))
  directory=prefix+record['relative_directory']+'/'
  rb=self.blob(commit,directory+'result.json');assert sha(rb)==record['result_hash'] and json.loads(rb)==result
  if not result.get('validated'):raise ValueError('unvalidated result')
  try:raw=self.blob(commit,directory+'predictions.csv.gz')
  except subprocess.CalledProcessError:return None
  assert sha(raw)==result['files']['predictions.csv.gz'],'prediction hash mismatch'
  assert result['protocol_hash']==self.ready['protocol_hash']
  assert result['input_hash']==self.ready['input_manifest_sha256']
  # Main/new-fold fits record exact executable file identities; R1 is an explicit alias.
  if 'resolved_config.json' in result.get('files',{}):
   cb=self.blob(commit,directory+'resolved_config.json');assert sha(cb)==result['files']['resolved_config.json'];cfg=json.loads(cb)
   assert all(cfg['source_files'].get(p)==h for p,h in self.ready['core_source_files'].items())
  p=pd.read_csv(io.BytesIO(raw),compression='gzip',dtype={'date':str,'asset_id':str,'origin_time':str,'endpoint_time':str})
  return p,result,{'owner':owner,'source_result_sha':commit,'job':id,'source_sha':result.get('source_sha'),'run_id':result.get('run_id'),'prediction_sha256':sha(raw),'source_path':directory+'predictions.csv.gz','protocol_hash':result['protocol_hash'],'input_hash':result['input_hash'],'split_hash':result['split_hash'],'transform_hash':result['transform_hash']}
 def load(self,model,seed,fold):
  owner='rtx3090' if model in ['UA','UC','SHARED_QUERY'] else 'a5000'
  job=('selected_'+model+'_42' if seed==42 else 'main_'+model+'_'+str(seed)+'_selected') if fold=='main' else (fold+'_'+model+'_'+str(seed) if fold=='R1' else fold+'_'+model+'_'+str(seed)+'_selected')
  item=self.resolve(owner,job)
  if item is None:self.missing.append({'model':model,'seed':seed,'fold':fold,'job':job,'reason':'exact selected saved prediction not yet published'});return
  frame,result,provenance=item;expected=self.expected(fold)
  assert not frame.duplicated(KEYS).any() and len(frame)==len(expected)
  frame=frame.sort_values(KEYS).reset_index(drop=True)
  cols=KEYS+['endpoint_time','legacy_label']+POPS
  assert frame[cols].equals(expected[cols]),'row/key/label/mask mismatch'
  assert frame.seed.eq(seed).all() and frame.model.eq(model).all()
  p=frame[PROBS].to_numpy(np.float64);assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all();s=p.sum(1);assert (s>0).all()
  error=float(np.max(abs(s-1)));assert error<1e-5,'large probability sum error'
  provenance.update(rows=len(frame),row_matches='exact full expected support',max_probability_sum_error=error)
  self.provenance.append(provenance);self.frames[model,seed,fold]=(frame,result)
 def run(self,out):
  for owner in ['rtx3090','a5000']:self.receive(owner)
  for model in MODELS:
   for fold in (['main','R1','R2','R3'] if model in ['UA','UC','UM_V2_DEPTH'] else ['main']):
    for seed in SEEDS:self.load(model,seed,fold)
  mets=[];paired=[];assets=[];groups=[];calibration=[]
  for (model,seed,fold),(frame,result) in self.frames.items():
   for split,base in frame.groupby('split'):
    for pop in POPS:
     f=base[base[pop].eq(1)]
     rec={'namespace':'new_P3_training','model':model,'seed':seed,'fold':fold,'split':split,'population':pop,'dates':f.date.nunique(),'assets':f.asset_id.nunique(),'retained_fraction':len(f)/len(base),**self.metric(f)};mets.append(rec)
     _,p,_=self.losses(f);confidence=p.max(1);correct=p.argmax(1)==f.legacy_label.to_numpy();bins=np.minimum((confidence*15).astype(int),14);ece=0
     for b in range(15):
      keep=bins==b;n=int(keep.sum());acc=float(correct[keep].mean()) if n else None;conf=float(confidence[keep].mean()) if n else None
      if n:ece+=n/len(f)*abs(acc-conf)
      calibration.append({'model':model,'seed':seed,'fold':fold,'split':split,'population':pop,'bin':b,'rows':n,'accuracy':acc,'mean_confidence':conf})
     rec['ece_15']=ece
  for comparator in ['UC','UM_V2_DEPTH','UX','U1','GRU_U1','SHARED_QUERY','TLOB_ADAPTED']:
   for fold_group in [['main']]+([['R1'],['R2'],['R3'],['R1','R2','R3']] if comparator in ['UC','UM_V2_DEPTH'] else []):
    absent=[(m,s,f) for f in fold_group for s in SEEDS for m in ['UA',comparator] if (m,s,f) not in self.frames]
    if absent:self.missing.append({'comparison':'UA-'+comparator,'folds':fold_group,'missing_model_seed_folds':absent});continue
    aa=[];bb=[]
    for fold in fold_group:
     for seed in SEEDS:
      a,ra=self.frames['UA',seed,fold];b,rb=self.frames[comparator,seed,fold]
      for key in ['protocol_hash','input_hash','split_hash','transform_hash','training_hash']:assert ra[key]==rb[key],key+' pair mismatch'
      assert a[KEYS+['legacy_label']].equals(b[KEYS+['legacy_label']]);aa.append(a);bb.append(b)
    a=pd.concat(aa,ignore_index=True);b=pd.concat(bb,ignore_index=True);name=fold_group[0] if len(fold_group)==1 else 'pooled_rolling'
    for split in (['validation','development_holdout'] if name=='main' else ['rolling_evaluation']):
     for pop in POPS:
      left=a[a.split.eq(split)&a[pop].eq(1)].sort_values(KEYS+['seed']).reset_index(drop=True);right=b[b.split.eq(split)&b[pop].eq(1)].sort_values(KEYS+['seed']).reset_index(drop=True)
      assert left[KEYS+['seed','legacy_label']].equals(right[KEYS+['seed','legacy_label']]);delta=self.losses(left)[0]-self.losses(right)[0]
      rec={'comparison':'UA-'+comparator,'fold':name,'split':split,'population':pop,'seed':'mean_3','rows':len(left)//3,'rows_including_seeds':len(left),'delta_nll':float(delta.mean()),'matched_seeds':'42,7,123'}
      if comparator in ['UC','UM_V2_DEPTH']:rec.update(self.bootstrap(left,right))
      paired.append(rec)
      left=left.assign(delta=delta)
      for seed,part in left.groupby('seed'):paired.append({'comparison':'UA-'+comparator,'fold':name,'split':split,'population':pop,'seed':int(seed),'rows':len(part),'delta_nll':float(part.delta.mean())})
      for asset,part in left.groupby('asset_id'):assets.append({'comparison':'UA-'+comparator,'fold':name,'split':split,'population':pop,'asset_id':asset,'rows':len(part)//3,'delta_nll':float(part.delta.mean())})
      for col in ['origin_bucket','relative_spread_bin','l5_depth_bin']:
       assert left[col].reset_index(drop=True).equals(right[col].reset_index(drop=True)),col+' differs across models'
       for value,part in left.groupby(col):groups.append({'comparison':'UA-'+comparator,'fold':name,'split':split,'population':pop,'grouping':col,'group':value,'rows':len(part)//3,'delta_nll':float(part.delta.mean()),'down':int(part.legacy_label.eq(0).sum())//3,'flat':int(part.legacy_label.eq(1).sum())//3,'up':int(part.legacy_label.eq(2).sum())//3})
  out.mkdir(parents=True,exist_ok=True)
  for name,rows,fields in [('metrics',mets,['model','seed','fold','split','population','rows','nll']),('paired_differences',paired,['comparison','fold','split','population','seed','delta_nll']),('per_asset_differences',assets,['comparison','asset_id','delta_nll']),('subgroup_differences',groups,['comparison','grouping','group','delta_nll']),('calibration',calibration,['model','seed','fold','split','population','bin','rows'])]:write_table(out/(name+'.csv'),rows,fields)
  manifest={'status':'partial' if self.missing else 'complete','namespace':'new_P3_training','data_handoff_sha':DATA_SHA,'protocol_hash':self.ready['protocol_hash'],'source_result_shas':self.refs,'verified_saved_prediction_files':len(self.frames),'provenance':self.provenance,'missing':self.missing,'bootstrap':{'draws':10000,'block_dates':5,'seed':20261001,'whole_date_panels_and_seeds':True,'rolling_blocks_preserve_folds':True},'training_or_inference_run_by_auditor':False,'interpretation':'Retrospective conditional-on-selection diagnostics. Future endpoint P3 validity is ex post, never a trading filter. No probability ensemble; missing outputs are not zero.'}
  dump(out/'analysis_manifest.json',manifest)
  text='# Independently matched native results\n\nStatus: '+manifest['status']+'\n\nExact advertised source result commits: '+json.dumps(self.refs,sort_keys=True)+'\n\nVerified selected saved files: '+str(len(self.frames))+'. Every loaded file passed its Git/result digest, full expected support, labels, masks, class order and probability checks.\n\n'+manifest['interpretation']+'\n\nMissing model/seed/fold identities are listed in analysis_manifest.json. Historical replay is separately frozen at dee398fb13078937833329a49f0665eb3c459b47.\n'
  tmp=out/'progress_report.tmp';tmp.write_text(text);tmp.replace(out/'progress_report.md')
  print(json.dumps({'status':manifest['status'],'source_result_shas':self.refs,'verified_files':len(self.frames),'missing_entries':len(self.missing)}))
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();Audit(json.loads(a.config.read_text())).run(a.output)
if __name__=='__main__':main()
