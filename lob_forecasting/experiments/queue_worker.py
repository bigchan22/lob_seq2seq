import os,subprocess,time
from pathlib import Path
from .queue_gates import validate_gpu

def gpu_snapshot(gpu,min_free_mb=8000):
 validate_gpu(gpu)
 query=['nvidia-smi','-i',str(gpu),'--query-gpu=name,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits']
 info=subprocess.run(query,capture_output=True,text=True);apps=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
 if info.returncode:return False,{'error':info.stderr.strip()}
 name,total,free,util=[x.strip() for x in info.stdout.split(',')];busy=bool(apps.stdout.strip())
 return (not busy and int(free)>=min_free_mb),{'physical_gpu':gpu,'name':name,'total_mb':int(total),'free_mb':int(free),'utilization':int(util),'active_processes':apps.stdout.strip()}

def run_worker(db,gpu,poll=30,once=False):
 validate_gpu(gpu)
 while True:
  ready,snap=gpu_snapshot(gpu)
  if not ready:
   if once:return None
   time.sleep(poll);continue
  job=db.claim('gpu',gpu)
  if not job:
   if once:return None
   time.sleep(poll);continue
  env={**os.environ,'CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUDA_VISIBLE_DEVICES':str(gpu),'QF_PHYSICAL_GPU':str(gpu),'QF_LOCAL_DEVICE':'cuda:0'}
  db.state(job['job_id'],'RUNNING',start_time=time.time(),heartbeat_time=time.time())
  p=subprocess.Popen(job['command'],shell=True,cwd=Path(__file__).resolve().parents[2],env=env)
  db.state(job['job_id'],'RUNNING',failure_message=str(p.pid))
  while p.poll() is None:db.heartbeat(job['job_id']);time.sleep(15)
  state='SUCCEEDED' if p.returncode==0 else 'FAILED';db.state(job['job_id'],state,return_code=p.returncode,finish_time=time.time(),failure_category='' if p.returncode==0 else 'deterministic_code')
  if state=='SUCCEEDED' and job['phase']=='smoke':
   smoke=[r for r in db.rows() if r['phase']=='smoke']
   if smoke and all(r['state']=='SUCCEEDED' for r in smoke):db.set_gate('model_smoke_pass',1,'all real-subset GPU smoke jobs succeeded');db.refresh()
  if once:return job
