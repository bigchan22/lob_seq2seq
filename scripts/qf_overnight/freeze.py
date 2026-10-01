"""Publishable READY from actual passing checks; never embeds its own commit SHA."""
import argparse,shutil
from common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--gates',required=True);p.add_argument('--cache',required=True);p.add_argument('--tlob-gates');p.add_argument('--tlob-block-reason');a=p.parse_args()
    result=read(Path(a.gates)/'gates.json');assert result['passed']
    required={'U1','UC','UA','UM_V2_DEPTH','UX','GRU_U1','SHARED_QUERY','U0','S'}
    assert required<=set(result['smokes'])
    protocol=read(DOCS/'protocol.json');assert digest(protocol)==result['protocol_hash']
    data=read(Path(a.cache)/'identity.json');core=source_files()
    assert result['source_files']==core,'Source changed after gates: revalidate the affected implementation before READY.'
    if READY.exists():
        previous=read(READY)
        assert previous['protocol_hash']==digest(protocol)
        assert previous['core_source_files']==core,'Core changed: a new reviewed protocol/source invalidation is required'
    write(DOCS/'core_gates.json',result)
    optional=False;optional_reason='Optional adapter effort pending; core is independent.';optional_files={}
    if a.tlob_gates:
        gate=read(Path(a.tlob_gates)/'gates.json');assert gate['passed'] and 'TLOB_ADAPTED' in gate['smokes']
        write(DOCS/'tlob_gates.json',gate);optional=True;optional_reason=None;optional_files=source_files(True)
    if a.tlob_block_reason:optional_reason=a.tlob_block_reason
    record={'run_group':GROUP,'protocol_hash':digest(protocol),'READY_CORE':True,'TLOB_ADAPTER_READY':optional,
            'TLOB_status':'READY' if optional else 'BLOCKED' if a.tlob_block_reason else 'PENDING_BOUNDED_ADAPTER',
            'TLOB_reason':optional_reason,'prior_code_commit':code_sha(),'published_at':utc(),
            'core_source_files':core,'core_executable_digest':digest(core),'optional_source_files':optional_files,
            'prepared_data_hashes':data['files'],'prepared_identity_sha256':sha(Path(a.cache)/'identity.json'),
            'input_manifest_sha256':data['input_manifest_sha256'],'core_gates_sha256':sha(DOCS/'core_gates.json'),
            'manifest_sha256':sha(CONFIG/'queue_manifest.json'),'split_manifest_sha256':sha(CONFIG/'splits.json'),
            'smoke_parameters':result['parameters'],'global_job_counts':read(CONFIG/'queue_manifest.json')['counts_by_owner'],
            'data_handoff_sha':HANDOFF,'consumer_rule':'Obtain READY_SHA from advertised Git ref externally. Execute only a clean immutable worktree; verify runtime_gate before claims.'}
    write(READY,record);print(json.dumps({k:record[k] for k in ['protocol_hash','READY_CORE','TLOB_ADAPTER_READY','core_executable_digest']}))

if __name__=='__main__':main()
