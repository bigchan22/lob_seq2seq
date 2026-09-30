"""Stage-2 queue CLI using a database separate from the completed pilot."""
import argparse,os,subprocess,time
from pathlib import Path
from .queue_db import QueueDB
from .queue_gates import validate_gpu
from .queue_worker import run_worker
from .discrimination_plan import build_plan
def db_path():return Path(os.environ['QF_DISC_ROOT'])/'queue/qf_discrimination.sqlite'
def main():
 p=argparse.ArgumentParser();p.add_argument('--db',type=Path,default=None);s=p.add_subparsers(dest='action',required=True)
 s.add_parser('init');s.add_parser('validate');s.add_parser('status');w=s.add_parser('worker');w.add_argument('--gpu',type=int,required=True);s.add_parser('coordinator');s.add_parser('pause');s.add_parser('resume');e=s.add_parser('export');e.add_argument('--output',type=Path,required=True)
 a=p.parse_args();db=QueueDB(a.db or db_path());db.initialize()
 if a.action=='init':db.add_jobs(build_plan()['jobs']);db.db.execute("INSERT OR REPLACE INTO metadata VALUES('paused','0')");db.db.commit();print(len(db.rows()))
 elif a.action=='validate':
  rows=db.rows();assert len(rows)==len({r['job_id'] for r in rows});assert all(r['physical_gpu'] in ('','0','1') for r in rows);assert not any('rolling' in r['phase'].lower() or 'lockbox' in r['phase'].lower() for r in rows);print('valid',len(rows))
 elif a.action=='status':
  rows=db.rows();print(' '.join(f'{x}={sum(r["state"]==x for r in rows)}' for x in ('PENDING','BLOCKED','CLAIMED','RUNNING','SUCCEEDED','FAILED','SKIPPED')))
  for g in ('0','1'):
   r=next((r for r in rows if r['physical_gpu']==g and r['state']=='RUNNING'),None);print('GPU',g,r['job_id'] if r else 'idle')
 elif a.action=='worker':validate_gpu(a.gpu);run_worker(db,a.gpu)
 elif a.action=='coordinator':
  while True:
   paused=db.db.execute("SELECT value FROM metadata WHERE key='paused'").fetchone()
   if not paused or paused[0]!='1':
    db.refresh();j=db.claim('cpu')
    if j:
     db.state(j['job_id'],'RUNNING',start_time=time.time());rc=subprocess.run(j['command'],shell=True,cwd=Path(__file__).resolve().parents[2],env={**os.environ}).returncode;db.state(j['job_id'],'SUCCEEDED' if rc==0 else 'FAILED',return_code=rc,finish_time=time.time(),failure_category='' if rc==0 else 'deterministic_code')
   time.sleep(5)
 elif a.action in ('pause','resume'):db.db.execute("INSERT OR REPLACE INTO metadata VALUES('paused',?)",('1' if a.action=='pause' else '0',));db.db.commit()
 elif a.action=='export':db.export(a.output)
if __name__=='__main__':main()
