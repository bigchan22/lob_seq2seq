"""SQLite WAL experiment queue with atomic claims and visible failures."""
from __future__ import annotations
import csv,json,os,sqlite3,time
from pathlib import Path
STATES=("BLOCKED","PENDING","CLAIMED","RUNNING","SUCCEEDED","FAILED","SKIPPED","CANCELLED")
FIELDS=("job_id","experiment_id","phase","model","asset_shard","seed","fold","horizon","representation","protocol","feature_set","resource_type","physical_gpu","priority","dependencies","gates","command","run_directory","state","creation_time","claim_time","start_time","heartbeat_time","finish_time","return_code","retry_count","max_retries","failure_category","failure_message","git_commit","resolved_config_hash","data_manifest_hash")

class QueueDB:
 def __init__(self,path):self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);self.db=sqlite3.connect(self.path,timeout=30);self.db.row_factory=sqlite3.Row;self.db.execute("PRAGMA journal_mode=WAL")
 def initialize(self):
  self.db.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
  self.db.execute("CREATE TABLE IF NOT EXISTS gates(name TEXT PRIMARY KEY,passed INTEGER NOT NULL DEFAULT 0,details TEXT,updated REAL)")
  cols=",".join(f"{x} TEXT" for x in FIELDS);self.db.execute(f"CREATE TABLE IF NOT EXISTS jobs({cols},CHECK(state IN {STATES}),CHECK(physical_gpu IS NULL OR physical_gpu IN ('','0','1')),CHECK(feature_set='no_investor' OR phase='secondary'))")
  self.db.execute("CREATE UNIQUE INDEX IF NOT EXISTS jobs_id ON jobs(job_id)");self.db.commit()
 def add_jobs(self,jobs):
  now=str(time.time());q=','.join('?'*len(FIELDS))
  for j in jobs:
   j={**{x:"" for x in FIELDS},**j};j['creation_time']=j['creation_time'] or now;j['dependencies']=json.dumps(j.get('dependencies',[])) if not isinstance(j.get('dependencies'),str) else j['dependencies'];j['gates']=json.dumps(j.get('gates',[])) if not isinstance(j.get('gates'),str) else j['gates'];j['retry_count']=str(j.get('retry_count',0));j['max_retries']=str(j.get('max_retries',0));self.db.execute(f"INSERT OR IGNORE INTO jobs VALUES({q})",[j[x] for x in FIELDS])
  self.db.commit()
 def set_gate(self,name,passed,details=""):self.db.execute("INSERT INTO gates VALUES(?,?,?,?) ON CONFLICT(name) DO UPDATE SET passed=excluded.passed,details=excluded.details,updated=excluded.updated",(name,int(passed),details,time.time()));self.db.commit()
 def refresh(self):
  passed={r['name'] for r in self.db.execute("SELECT * FROM gates WHERE passed=1")};success={r['job_id'] for r in self.db.execute("SELECT job_id FROM jobs WHERE state='SUCCEEDED'")}
  for r in self.db.execute("SELECT * FROM jobs WHERE state='BLOCKED'").fetchall():
   if set(json.loads(r['dependencies'] or '[]'))<=success and set(json.loads(r['gates'] or '[]'))<=passed:self.db.execute("UPDATE jobs SET state='PENDING' WHERE job_id=?",(r['job_id'],))
  self.db.commit()
 def claim(self,resource,gpu=None):
  if gpu is not None and str(gpu) not in {'0','1'}:raise ValueError("only physical GPUs 0 and 1 are allowed")
  self.refresh();self.db.execute("BEGIN IMMEDIATE")
  sql="SELECT * FROM jobs WHERE state='PENDING' AND resource_type=?";args=[resource]
  if resource=='gpu':sql+=" AND (physical_gpu='' OR physical_gpu=?)";args.append(str(gpu))
  r=self.db.execute(sql+" ORDER BY CAST(priority AS INTEGER) DESC,job_id LIMIT 1",args).fetchone()
  if r:self.db.execute("UPDATE jobs SET state='CLAIMED',physical_gpu=?,claim_time=?,heartbeat_time=? WHERE job_id=? AND state='PENDING'",(str(gpu) if gpu is not None else '',time.time(),time.time(),r['job_id']))
  self.db.commit();return dict(r) if r else None
 def state(self,job,state,**values):
  if state not in STATES:raise ValueError(state)
  sets=['state=?'];args=[state]
  for k,v in values.items():sets.append(f"{k}=?");args.append(v)
  args.append(job);self.db.execute(f"UPDATE jobs SET {','.join(sets)} WHERE job_id=?",args);self.db.commit()
 def heartbeat(self,job):self.db.execute("UPDATE jobs SET heartbeat_time=? WHERE job_id=?",(time.time(),job));self.db.commit()
 def recover_stale(self,age=900):
  fixed=[]
  for r in self.db.execute("SELECT * FROM jobs WHERE state IN ('CLAIMED','RUNNING') AND CAST(heartbeat_time AS REAL)<?",(time.time()-age,)):
   pid=int(r['failure_message']) if (r['failure_message'] or '').isdigit() else 0
   if pid and Path(f'/proc/{pid}').exists():continue
   self.state(r['job_id'],'PENDING',failure_category='recovered_stale');fixed.append(r['job_id'])
  return fixed
 def rows(self):return [dict(x) for x in self.db.execute("SELECT * FROM jobs ORDER BY phase,CAST(priority AS INTEGER) DESC,job_id")]
 def export(self,path):
  rows=self.rows();path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
 def close(self):self.db.close()
