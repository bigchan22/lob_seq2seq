import argparse,csv,json,os,subprocess,time
from pathlib import Path
from .queue_db import QueueDB
from .queue_gates import validate_gpu
from .queue_worker import run_worker
from .qf_plan import build_plan

def default_db():return Path(os.environ.get('QF_ARTIFACT_ROOT','artifacts'))/'queue/qf_queue.sqlite'
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--db',type=Path,default=default_db());s=p.add_subparsers(dest='action',required=True)
 s.add_parser('init').add_argument('--plan');s.add_parser('validate');s.add_parser('status');l=s.add_parser('list');l.add_argument('--phase');w=s.add_parser('worker');w.add_argument('--gpu',type=int,required=True);w.add_argument('--once',action='store_true');s.add_parser('coordinator');s.add_parser('pause');s.add_parser('resume');rf=s.add_parser('retry-failed');rf.add_argument('--job-id',required=True);rs=s.add_parser('recover-stale');rs.add_argument('--age',type=int,default=900);e=s.add_parser('export');e.add_argument('--output',type=Path,default=Path('queue_plan.csv'))
 a=p.parse_args(argv);db=QueueDB(a.db);db.initialize()
 if a.action=='init':db.add_jobs(build_plan()['jobs']);db.db.execute("INSERT OR REPLACE INTO metadata VALUES('paused','0')");db.db.commit();print(f'initialized {len(db.rows())} jobs at {a.db}')
 elif a.action=='validate':
  rows=db.rows();assert len({x['job_id'] for x in rows})==len(rows);assert all(x['physical_gpu'] in {'','0','1'} for x in rows);assert sum(x['phase']=='confirmatory' and x['resource_type']=='gpu' for x in rows)==72;assert all(x['feature_set']=='no_investor' for x in rows);print('queue valid')
 elif a.action=='status':
  rows=db.rows();counts={x:sum(r['state']==x for r in rows) for x in ['PENDING','BLOCKED','CLAIMED','RUNNING','SUCCEEDED','FAILED']};print(' '.join(f'{k}={v}' for k,v in counts.items()))
  for gpu in ['0','1']:
   active=next((r for r in rows if r['physical_gpu']==gpu and r['state']=='RUNNING'),None);print(f'GPU {gpu}: '+(f"{active['job_id']} epoch={active.get('failure_message','')} heartbeat={active['heartbeat_time']}" if active else 'idle'))
  failed=[r['job_id'] for r in rows if r['state']=='FAILED'];blocked=[r['job_id'] for r in rows if r['state']=='BLOCKED'];print('failed:',','.join(failed) or 'none');print('blocked:',len(blocked));subprocess.run(['df','-h',str(a.db.parent)])
 elif a.action=='list':
  for r in db.rows():
   if not a.phase or r['phase']==a.phase:print(r['job_id'],r['state'],r['model'])
 elif a.action=='worker':validate_gpu(a.gpu);run_worker(db,a.gpu,once=a.once)
 elif a.action=='coordinator':
  while True:
   paused=db.db.execute("SELECT value FROM metadata WHERE key='paused'").fetchone();
   if not paused or paused[0]!='1':db.refresh();job=db.claim('cpu')
   if job:db.state(job['job_id'],'RUNNING',start_time=time.time());rc=subprocess.run(job['command'],shell=True).returncode;db.state(job['job_id'],'SUCCEEDED' if rc==0 else 'FAILED',return_code=rc,finish_time=time.time())
   time.sleep(15)
 elif a.action in {'pause','resume'}:db.db.execute("INSERT OR REPLACE INTO metadata VALUES('paused',?)",('1' if a.action=='pause' else '0',));db.db.commit();print(a.action+'d')
 elif a.action=='retry-failed':
  r=db.db.execute("SELECT * FROM jobs WHERE job_id=?",(a.job_id,)).fetchone();
  if not r or r['state']!='FAILED':raise SystemExit('job is not failed')
  if r['failure_category']!='transient_infrastructure' or int(r['retry_count'])>=int(r['max_retries']):raise SystemExit('retry prohibited')
  db.state(a.job_id,'PENDING',retry_count=str(int(r['retry_count'])+1))
 elif a.action=='recover-stale':print('\n'.join(db.recover_stale(a.age)))
 elif a.action=='export':db.export(a.output);print(a.output)
if __name__=='__main__':main()
