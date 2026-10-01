"""Add a byte-hashed receiver schema projection without changing the protocol."""
import sys,ast
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from receiver_bridge.bridge import GATES

def update():
    ready=read(READY);protocol=read(DOCS/'protocol.json');assert digest(protocol)==ready['protocol_hash']
    projection=dict(protocol,run_group=GROUP,data_handoff_sha=HANDOFF,training_population='P3',validation_population='P3',
                    selection_seeds=SEEDS,learning_rates=LRS,canonical_protocol_path=str((DOCS/'protocol.json').relative_to(ROOT)),
                    canonical_protocol_hash=digest(protocol),schema_projection='receiver v1; aliases only, no scientific changes')
    path=DOCS/'receiver_protocol.json';write(path,projection)
    ready.update(protocol_path=str(path.relative_to(ROOT)),protocol_sha256=sha(path),input_sha='073cd89b556224eeae22413c3c05188a947e990a',
                 owner_models={'rtx3090':['UA','UC','SHARED_QUERY'],'a5000':['U1','UM_V2_DEPTH','UX','GRU_U1','U0','TLOB_ADAPTED','S']},
                 split_manifest_path=str((CONFIG/'splits.json').relative_to(ROOT)),job_manifest_path=str((CONFIG/'queue_manifest.json').relative_to(ROOT)))
    source=source_files();critical={p:h for p,h in source.items() if p.endswith(('common.py','models.py','data.py','trainer.py','qf_variants.py','transforms.py','discrimination_factors.py','build_data_review.py'))}
    tree=ast.parse((ROOT/'scripts/qf_overnight/scoring.py').read_text())
    critical['scoring_metric_functions_ast']=digest([ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['normalized','losses','metrics','paired_bootstrap']])
    ready['comparison_contract_files']=critical;ready['comparison_contract_digest']=digest(critical)
    runner='scripts/qf_overnight/receiver_bridge/bridge.py';base=['{python}',runner]
    ready['receiver']={'verification_gates':GATES,'commands':{
        'verify':base+['verify','--input-root','{input_root}','--data-root','{data_root}','--run-root','{run_root}','--owner','{owner}'],
        'smoke':base+['smoke','--run-root','{run_root}','--owner','{owner}'],
        'launch':base+['run','--run-root','{run_root}','--owner','{owner}','--workers','{workers}','--control-path','{control_path}'],
        'pause':base+['pause','--run-root','{run_root}'],'resume':base+['resume','--run-root','{run_root}'],
        'retry':base+['retry','--run-root','{run_root}','--job','{job_id}']}}
    files=set(source)
    for directory in [ROOT/'scripts/qf_overnight',CONFIG,DOCS]:
        files.update(str(p.relative_to(ROOT)) for p in directory.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.csv','.md','.txt'] and '__pycache__' not in p.parts)
    files.update(str(p.relative_to(ROOT)) for p in (ROOT/'scripts/qf_overnight/vendor_tlob').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    ready['files']=[{'path':p,'sha256':sha(ROOT/p)} for p in sorted(files)]
    ready['core_source_files']=source;ready['core_executable_digest']=digest(source)
    ready['transport_adapter_digest']=digest({f['path']:f['sha256'] for f in ready['files'] if '/receiver_bridge/' in f['path']})
    ready['published_at']=utc();ready['prior_code_commit']=code_sha();write(READY,ready)
    print(digest(protocol),ready['protocol_sha256'],ready['core_executable_digest'])
if __name__=='__main__':update()
