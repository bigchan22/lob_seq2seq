import json,os,platform,subprocess,sys,tempfile
from pathlib import Path
class RunDirectory:
 def __init__(self,root,experiment_id,run_id,overwrite=False):
  self.path=Path(root)/"runs"/experiment_id/run_id
  if self.path.exists() and not overwrite: raise FileExistsError(self.path)
  for d in ["config","data","logs","checkpoints","predictions","metrics"]: (self.path/d).mkdir(parents=True,exist_ok=True)
 def write_text(self,name,value): (self.path/name).write_text(value)
 def save_config(self,requested,resolved):
  import yaml
  (self.path/"config/requested.yaml").write_text(yaml.safe_dump(requested,sort_keys=True)); (self.path/"config/resolved.json").write_text(json.dumps(resolved,indent=2,sort_keys=True))
 def record_provenance(self,command):
  self.write_text("command.txt",command+"\n"); self.write_text("environment.txt",f"python={sys.version}\nplatform={platform.platform()}\n")
  commit=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(); dirty=subprocess.run(["git","status","--short"],capture_output=True,text=True).stdout; self.write_text("git_state.txt",f"commit={commit}\n{dirty}")
 def write_status(self,status):
  target=self.path/"status.json"; fd,tmp=tempfile.mkstemp(dir=target.parent)
  try:
   with os.fdopen(fd,"w") as f: json.dump(status,f,indent=2); f.write("\n")
   os.replace(tmp,target)
  finally:
   if os.path.exists(tmp): os.unlink(tmp)
