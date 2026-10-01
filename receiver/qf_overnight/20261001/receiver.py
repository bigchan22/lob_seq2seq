"""Durable Git receiver/publisher. Scientific code is supplied only by A5000."""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

RUN_GROUP = 'qf-august-overnight-20261001'
OWNER = 'rtx3090'
REMOTE = 'git@github.com:bigchan22/lob_seq2seq.git'
PREFIX = Path('results/qf_overnight/20261001/rtx3090')
REQUIRED_GATES = ['labels_masks_splits', 'causality', 'masked_loss_gradient',
                  'own_only_invariance', 'parameter_counts',
                  'masked_accumulation_partial_window', 'train_only_scaling',
                  'r1_reuse_identity', 'restart_idempotency_stale_lease',
                  'real_data_smoke']


def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def read(path, default=None):
    try: return json.loads(Path(path).read_text())
    except FileNotFoundError: return default
def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temp, path)
def identity(pid):
    try:
        parts = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
        return None if parts[0] in ('Z', 'X') else parts[19]
    except (OSError, IndexError): return None
def alive(record):
    return bool(record and identity(record.get('pid', -1)) == record.get('start_ticks'))
def relative_path(path):
    p = Path(path)
    if p.is_absolute() or '..' in p.parts or not p.parts: raise ValueError('Unsafe relative path')
    return p


class Receiver:
    def __init__(self, config):
        self.config_path = str(Path(config).resolve()); self.c = read(config)
        self.root = Path(self.c['runtime_root']); self.root.mkdir(parents=True, exist_ok=True)
        self.pub = Path(self.c['publication_worktree']); self.bridge = Path(self.c['bridge'])
        self.state = read(self.root / 'receiver_status.json', {})
        self.state.setdefault('started_at', now())
        self.state.setdefault('deadline_epoch', time.time() + 12 * 3600)
        self.state.setdefault('phase', 'WAITING_FOR_READY')
        self.env = os.environ.copy()
        self.env.update(GIT_OPTIONAL_LOCKS='0', GIT_LFS_SKIP_SMUDGE='1', GIT_TERMINAL_PROMPT='0',
                        GIT_SSH_COMMAND='ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15',
                        PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                        OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
        self.last_publish = 0.; self.last_other = 0.; self.last_aggregate_signature = None

    def event(self, kind, **kw):
        with (self.root / 'events.jsonl').open('a') as f:
            f.write(json.dumps({'time': now(), 'event': kind, **kw}, sort_keys=True) + '\n')

    def save(self):
        self.state.update(updated_at=now(), receiver_pid=os.getpid(),
                          receiver_start_ticks=identity(os.getpid()), run_group=RUN_GROUP, owner=OWNER)
        atomic(self.root / 'receiver_status.json', self.state)

    def git(self, *args, where=None, check=True, timeout=120):
        p = subprocess.run(['git', '-C', str(where or self.bridge), *args],
                           env=self.env, capture_output=True, text=True, timeout=timeout)
        if check and p.returncode:
            # Never log the environment, ssh debugging, or arbitrary process arguments.
            raise RuntimeError('git %s failed (%d): %s' % (args[0], p.returncode, p.stderr[-1500:]))
        return p

    def advertised(self, ref):
        p = self.git('ls-remote', '--heads', 'origin', ref)
        records = [x.split() for x in p.stdout.splitlines()]
        return next((a for a, b in records if b == ref), None)

    def get_source(self, ref, commit):
        local_ref = 'refs/remotes/origin/' + ref.removeprefix('refs/heads/')
        self.git('fetch', '--filter=blob:none', '--depth=8', '--no-tags', '--no-recurse-submodules',
                 'origin', ref + ':' + local_ref)
        if self.git('rev-parse', local_ref).stdout.strip() != commit:
            raise ValueError('Ref moved during fetch; re-resolve on next poll instead of silently switching')
        return self.git('ls-tree', '-r', commit).stdout

    def get_ready(self, commit, tree):
        items = {}
        for line in tree.splitlines():
            meta, path = line.split('\t'); mode, kind, oid = meta.split()
            items[path] = (kind, oid, mode)
        wanted = self.c['ready_path']
        if wanted not in items: return None
        blob = self.git('cat-file', 'blob', commit + ':' + wanted).stdout
        ready = json.loads(blob)
        if ready.get('READY_CORE') is not True: return None
        return ready, items

    def inspect_ready(self, commit, ready, items):
        if ready.get('run_group') != RUN_GROUP: raise ValueError('READY run_group mismatch')
        if ready.get('data_handoff_sha') != self.c['data_sha']: raise ValueError('READY data SHA mismatch')
        if ready.get('input_sha') != self.c['input_sha']: raise ValueError('READY input lineage mismatch')
        if ready.get('owner_models', {}).get(OWNER) != ['UA', 'UC', 'SHARED_QUERY']:
            raise ValueError('READY static ownership missing/mismatched')
        files = ready.get('files', [])
        if not isinstance(files, list) or not files: raise ValueError('READY files digest list missing')
        verified_paths = {str(relative_path(f['path'])) for f in files}
        shared_files = {p for p in items if p.startswith(('scripts/qf_overnight/', 'configs/qf_overnight/'))
                        and p.endswith(('.py','.json','.yaml','.yml','.sh'))}
        if not shared_files.issubset(verified_paths): raise ValueError('READY omits shared source/config digests')
        for f in files:
            p = str(relative_path(f['path']))
            if p not in items or items[p][0] != 'blob' or items[p][2] == '120000':
                raise ValueError('Required source blob missing or symlink: ' + p)
            if not re.fullmatch('[0-9a-f]{64}', f['sha256']): raise ValueError('Invalid source digest')
        for key in ('protocol_path', 'split_manifest_path', 'job_manifest_path'):
            if ready.get(key) not in verified_paths: raise ValueError(key + ' not covered by READY digests')
        interface = ready.get('receiver', {})
        for command in ('verify', 'smoke', 'launch', 'pause', 'resume', 'retry'):
            argv = interface.get('commands', {}).get(command)
            if not isinstance(argv, list) or len(argv) < 2 or argv[0] != '{python}':
                raise ValueError('Expected receiver.commands.' + command + ' argv template')
            script = argv[1].removeprefix('{source}/')
            if not script.startswith('scripts/qf_overnight/') or script not in verified_paths:
                raise ValueError('Invocation script must be a hashed shared runner: ' + script)
        for gate in REQUIRED_GATES:
            if gate not in interface.get('verification_gates', []): raise ValueError('Missing gate ' + gate)
        return files

    def materialize(self, commit, ready, files, items):
        work = Path(self.c['source_root_parent']) / commit
        if work.exists():
            if self.git('rev-parse', 'HEAD', where=work).stdout.strip() != commit:
                raise ValueError('Execution path occupied by different source')
            if self.git('status', '--porcelain', where=work).stdout.strip():
                raise ValueError('Execution worktree dirty; preserve it')
        else:
            # Entire tree must be code/compact contract or already approved data-review payload.
            forbidden = [p for p in items if p.endswith(('.pt', '.pth', '.ckpt', '.sqlite', '.db'))]
            if forbidden: raise ValueError('Ref contains forbidden checkout payloads: ' + repr(forbidden[:8]))
            new_prefixes = ('scripts/qf_overnight/', 'configs/qf_overnight/',
                            'docs/qf_overnight/', 'coordination/qf_overnight/',
                            'tests/', 'lob_forecasting/', 'receiver/')
            base = self.git('ls-tree', '-r', self.c['data_sha']).stdout
            base_paths = {line.split('\t')[1] for line in base.splitlines()}
            unknown = [p for p in items if p not in base_paths and not p.startswith(new_prefixes)]
            if unknown: raise ValueError('Unexpected new tree paths require producer clarification: ' + repr(unknown[:8]))
            work.parent.mkdir(parents=True, exist_ok=True)
            self.git('worktree', 'add', '--no-checkout', '--detach', str(work), commit)
            self.git('config', '--worktree', 'core.sparseCheckout', 'false', where=work)
            self.git('read-tree', '-mu', 'HEAD', where=work, timeout=240)
        for f in files:
            p = work / relative_path(f['path'])
            if sha(p) != f['sha256']: raise ValueError('Executable-source hash mismatch: ' + f['path'])
        protocol = read(work / ready['protocol_path'])
        if sha(work / ready['protocol_path']) != ready.get('protocol_sha256'):
            raise ValueError('Protocol hash mismatch')
        # These identifiers are independently fixed by the user, never taken from performance.
        if protocol.get('run_group') != RUN_GROUP or protocol.get('data_handoff_sha') != self.c['data_sha']:
            raise ValueError('Protocol lineage mismatch')
        if protocol.get('training_population') != 'P3' or protocol.get('validation_population') != 'P3':
            raise ValueError('P3 training/selection contract missing')
        if protocol.get('selection_seeds') != [42, 7, 123]: raise ValueError('Seed contract mismatch')
        if protocol.get('learning_rates') != [0.00005, 0.0001, 0.0002, 0.0004]:
            raise ValueError('Bounded learning-rate contract mismatch')
        input_manifest = Path(self.c['data_worktree']) / 'recovery/input_manifest.csv'
        if ready.get('input_manifest_sha256') != sha(input_manifest): raise ValueError('Input manifest mismatch')
        return work

    def argv(self, ready, source, command):
        values = {'python': self.c['python'], 'source': str(source),
                  'input_root': self.c['input_root'], 'data_root': self.c['data_worktree'],
                  'run_root': str(self.root / 'jobs'), 'owner': OWNER,
                  'gpu_uuids': ','.join(self.state.get('active_gpu_uuids',self.c['gpu_uuids'])),
                  'workers': str(len(self.state.get('active_gpu_uuids',self.c['gpu_uuids']))),
                  'control_path': str(self.root / 'control.json'),
                  'other_results': str(self.root / 'other_results'),
                  'output_root': str(self.root / 'publish'),
                  'issue_root': str(self.root / 'issues'),
                  'job_id': self.state.get('retry_job', '')}
        argv = [str(a).format(**values) for a in ready['receiver']['commands'][command]]
        if not Path(argv[1]).is_absolute(): argv[1] = str(source / argv[1])
        Path(argv[1]).resolve().relative_to(source.resolve())
        return argv

    def command(self, ready, source, name, timeout=1200):
        log = self.root / ('gate_' + name + '_' + source.name[:12] + '.log')
        with log.open('ab') as f:
            p = subprocess.run(self.argv(ready, source, name), cwd=source, env=self.worker_env(),
                               stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
        self.event('command_finished', command=name, returncode=p.returncode, log=log.name)
        if p.returncode: raise RuntimeError(name + ' failed; see ' + log.name)

    def worker_env(self):
        e = self.env.copy()
        e.update(CUDA_VISIBLE_DEVICES=','.join(self.state.get('active_gpu_uuids',self.c['gpu_uuids'])),
                 QF_OWNER=OWNER, QF_RUN_GROUP=RUN_GROUP,
                 QF_DATA_ROOT=self.c['data_worktree'], QF_INPUT_ROOT=self.c['input_root'],
                 QF_RUN_ROOT=str(self.root / 'jobs'), QF_CONTROL_PATH=str(self.root / 'control.json'),
                 QF_CPU_THREADS='2', QF_DATALOADER_WORKERS='0')
        return e

    def issue(self, category, message, commit=None):
        key = hashlib.sha256((category + str(commit) + message).encode()).hexdigest()[:16]
        path = self.root / 'issues' / (key + '.json')
        if not path.exists():
            atomic(path, {'time': now(), 'category': category, 'source_sha': commit,
                          'owner': OWNER, 'message': message, 'action': 'A5000: publish corrected shared source/READY; receiver keeps polling. No local semantic patch.'})
            self.event('issue', category=category, issue=key)

    def resource_available(self):
        free = shutil.disk_usage(self.root).free
        if free < self.c['minimum_free_bytes']: return False, 'disk pressure'
        p = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=20)
        if p.returncode: return False, 'GPU allocation query failed'
        occupied = {line.split(',')[0].strip() for line in p.stdout.splitlines() if ',' in line}
        free_gpus = [g for g in self.c['gpu_uuids'] if g not in occupied]
        if not free_gpus: return False, 'all allocated GPUs occupied by existing processes'
        self.state['active_gpu_uuids'] = free_gpus
        return True, None

    def launch(self, ready, source):
        if alive(self.state.get('coordinator')): return
        ok, reason = self.resource_available()
        if not ok: self.state.update(phase='WAITING_FOR_RESOURCE', resource_blocker=reason); return
        gates_started=time.time()
        self.command(ready, source, 'verify')
        self.command(ready, source, 'smoke')
        gate_report = self.root / 'jobs' / 'receiver_gate_report.json'
        report = read(gate_report, {})
        if not gate_report.exists() or gate_report.stat().st_mtime < gates_started:
            raise ValueError('Gate report is absent or stale')
        if report.get('protocol_sha256') != ready['protocol_sha256'] or report.get('data_handoff_sha') != self.c['data_sha']:
            raise ValueError('Fresh executable gate report lacks pinned hashes')
        for gate in REQUIRED_GATES:
            if report.get('gates', {}).get(gate) is not True: raise ValueError('Gate not demonstrated: ' + gate)
        # Common coordinator is foreground and durable in its own process group.
        with (self.root / 'coordinator.log').open('ab') as log:
            p = subprocess.Popen(self.argv(ready, source, 'launch'), cwd=source,
                                 env=self.worker_env(), stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record = {'pid': p.pid, 'start_ticks': identity(p.pid), 'process_group': p.pid,
                  'started_at': now(), 'source_sha': source.name}
        self.state.update(coordinator=record, phase='RUNNING', launched_at=now())
        self.event('coordinator_launched', **record)

    def snapshot(self, owner, commit, tree):
        prefix='results/qf_overnight/20261001/'+owner+'/'
        out=self.root/'result_snapshots'/owner/commit
        if (out/'_receipt.json').exists(): return out
        out.mkdir(parents=True,exist_ok=True); copied=[]
        for line in tree.splitlines():
            meta,path=line.split('\t'); mode,kind,oid=meta.split()
            if not path.startswith(prefix) or kind!='blob' or mode=='120000': continue
            rel=relative_path(path[len(prefix):])
            if rel.parts[0] in ('consolidated','issues') or rel.suffix not in ('.json','.csv','.md'): continue
            size=int(self.git('cat-file','-s',oid).stdout)
            if size>20_000_000: continue
            payload=self.git('cat-file','blob',oid).stdout
            dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(payload)
            copied.append({'path':str(rel),'git_blob_oid':oid,'sha256':sha(dest),'bytes':dest.stat().st_size})
        atomic(out/'_receipt.json',{'source_result_sha':commit,'files':copied})
        return out

    def collect_other(self):
        ref=self.c['other_result_ref'];commit=self.advertised(ref)
        if not commit or commit==self.state.get('other_result_sha'): return
        tree=self.get_source(ref,commit);out=self.snapshot('a5000',commit,tree)
        self.state['other_result_sha']=commit;self.state['other_snapshot']=str(out)
        self.event('other_results_received',source_result_sha=commit)

    def aggregate(self):
        # Read immutable, already-published commits, never a live output directory.
        receipt=read(self.root/'publication_receipt.json',{})
        commit=receipt.get('sha');local=None;other=self.state.get('other_snapshot')
        if commit:
            tree=self.git('ls-tree','-r',commit).stdout
            local=self.snapshot(OWNER,commit,tree)
        signature=(commit,self.state.get('other_result_sha'))
        if signature==self.last_aggregate_signature: return
        script=Path(__file__).with_name('consolidate.py')
        if not script.exists(): return
        argv=[self.c['python'],'-B',str(script),'--data',self.c['data_worktree'],
              '--output',str(self.root/'consolidated'),'--cache',str(self.root/'expected_support.json')]
        if local: argv+=['--local',str(local)]
        if other: argv+=['--other',other]
        if self.state.get('protocol_hash'):argv+=['--protocol-hash',self.state['protocol_hash']]
        env=self.env.copy();env['CUDA_VISIBLE_DEVICES']=''
        with (self.root/'consolidation.log').open('ab') as log:
            result=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
        if result.returncode:
            self.issue('CONSOLIDATION_SCHEMA_OR_SUPPORT','See consolidation.log; no unverified comparisons published',self.state.get('ready_sha'))
        else:
            self.state['consolidation_status']=read(self.root/'consolidated/consolidation_manifest.json',{}).get('status')
            self.last_aggregate_signature=signature

    def other_terminal(self):
        root=self.state.get('other_snapshot')
        if not root:return False
        for path in (Path(root)/'queue_status.json',Path(root)/'runs/queue_status.json'):
            s=read(path,{})
            if s.get('all_terminal') is True or s.get('shared_queue_status',{}).get('all_terminal') is True:return True
        return False

    def final_publish(self):
        # A network failure cannot keep the 12-hour READY poll alive forever.
        for attempt in range(3):
            try:self.publish();return
            except Exception as error:
                self.state.update(last_error=str(error),unsynced=True);self.event('terminal_publication_failed',attempt=attempt+1,error=str(error));self.save()
                if attempt<2:time.sleep(15*(attempt+1))
        self.export_files()

    def export_files(self):
        dest = self.pub / PREFIX; dest.mkdir(parents=True, exist_ok=True)
        state = dict(self.state)
        for k in ('receiver_pid', 'receiver_start_ticks'): state.pop(k, None)
        state['local_runtime_alias'] = 'RECEIVER_RUNTIME'
        state['planned_owned_training_fits'] = 30
        state['planned_fits_are_not_materialized_jobs'] = not bool(self.state.get('ready_sha'))
        state['shared_queue_status'] = read(self.root / 'jobs/queue_status.json', {'counts': None, 'reason':'Common queue not launched yet'})
        atomic(dest / 'queue_status.json', state)
        if (self.root / 'events.jsonl').exists():
            shutil.copyfile(self.root / 'events.jsonl', dest / 'receiver_events.jsonl')
        for path in (self.root / 'issues').glob('*.json') if (self.root / 'issues').exists() else []:
            target = dest / 'issues' / path.name; target.parent.mkdir(exist_ok=True); shutil.copyfile(path,target)
        exports = read(self.root / 'jobs/publication_manifest.json', {'files':[]})
        for entry in exports['files']:
            rel = relative_path(entry['path'])
            if rel.suffix in ('.pt','.pth','.ckpt','.sqlite','.db') or 'checkpoint' in rel.name.lower():
                raise ValueError('Checkpoint/live database export refused')
            src = self.root / 'jobs' / rel; src.resolve().relative_to((self.root/'jobs').resolve())
            if src.is_symlink() or src.stat().st_size > 90_000_000: raise ValueError('Unsafe/oversized publication payload')
            if sha(src) != entry['sha256']: raise ValueError('Export digest changed before snapshot')
            target = dest / 'runs' / rel; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src,target)
            if sha(target) != entry['sha256']: raise ValueError('Export was not immutable during copy')
        for path in (self.root/'consolidated').glob('*') if (self.root/'consolidated').exists() else []:
            if path.is_file() and path.suffix in ('.json','.csv','.md'):
                target=dest/'consolidated'/path.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(path,target)
        counts = state['shared_queue_status'].get('counts')
        report = '# RTX3090 overnight progress\n\n'
        report += 'Run group: '+RUN_GROUP+'\n\nPhase: '+state['phase']+'\n\n'
        report += 'READY SHA: '+str(state.get('ready_sha'))+'; protocol SHA256: '+str(state.get('protocol_hash'))+'\n\n'
        report += 'Actual shared queue counts: '+json.dumps(counts)+'\n\n'
        report += 'Planned ownership: 18 main UA/UC/SHARED_QUERY fits + 12 rolling UA/UC fits, plus finite inference/reuse/analysis nodes. These are not reported as launched jobs until the common DAG exists.\n\n'
        report += 'Historical replay is separate; see historical_replay_reference.json. No performance-based launch gate.\n\n'
        report += 'Consolidated analysis status: '+str(state.get('consolidation_status','pending exports'))+'; see consolidated/progress_report.md.\n\n'
        report += 'Other owner result SHA: '+str(state.get('other_result_sha'))+'\n\n'
        report += 'Current blocker: '+str(state.get('last_error') or state.get('resource_blocker') or ('common READY not published' if not state.get('ready_sha') else None))+'\n\n'
        report += 'Status is partial until every finite job is terminal and matching required outputs validate. Provider timing unknown; retrospective development data; future endpoint P3 eligibility is not a trading or peer-availability rule.\n'
        (dest/'progress_report.md').write_text(report)
        if self.state['phase'] in ('COMPLETE','TERMINAL_PARTIAL','BLOCKED_READY_TIMEOUT','STOPPED'):
            (dest/'final_report.md').write_text(report)
        if time.time() >= self.state['deadline_epoch'] - 4*3600 and not (dest/'eight_hour_progress.md').exists():
            (dest/'eight_hour_progress.md').write_text(report)

    def publish(self):
        lock = (self.root/'publication.lock').open('a+')
        fcntl.flock(lock,fcntl.LOCK_EX)
        self.export_files()
        names = []
        for prefix in [PREFIX, Path('receiver/qf_overnight/20261001')]:
            names.extend(str(p.relative_to(self.pub)) for p in (self.pub/prefix).rglob('*') if p.is_file())
        for name in sorted(names): self.git('add', '--', name, where=self.pub)
        changed = self.git('diff','--cached','--name-only',where=self.pub).stdout.splitlines()
        if any(not p.startswith((str(PREFIX)+'/', 'receiver/qf_overnight/20261001/')) for p in changed):
            raise ValueError('Unrelated staged paths; publisher stopped')
        if changed:
            staged=self.git('diff','--cached','--numstat',where=self.pub).stdout
            self.event('staged_review',files=changed,numstat=staged)
            for path in changed:
                f=self.pub/relative_path(path)
                if f.stat().st_size>90_000_000 or f.suffix in ('.sqlite','.db','.pt','.pth','.ckpt'):
                    raise ValueError('Forbidden staged file')
            self.git('-c','core.hooksPath=/dev/null','-c','commit.gpgsign=false','commit','-m',
                     'Snapshot RTX3090 overnight receiver and owned results',where=self.pub)
        commit = self.git('rev-parse','HEAD',where=self.pub).stdout.strip()
        self.git('-c','core.hooksPath=/dev/null','push','origin','HEAD:refs/heads/'+self.c['result_branch'],where=self.pub)
        actual = self.advertised('refs/heads/'+self.c['result_branch'])
        if actual != commit: raise RuntimeError('Push outcome not advertised as expected')
        atomic(self.root/'publication_receipt.json', {'time':now(), 'branch':self.c['result_branch'], 'sha':commit, 'verified':True})
        self.state['last_published_result_sha'] = commit
        self.state.pop('unsynced',None)

    def controls(self, ready, source):
        ctl = read(self.root/'control.json',{})
        if ctl.get('generation') == self.state.get('control_generation'):
            return 'stop' if ctl.get('stop') else ('pause' if ctl.get('paused') else None)
        if ready and source and alive(self.state.get('coordinator')):
            if ctl.get('paused') or ctl.get('stop'): self.command(ready,source,'pause',timeout=60)
            elif ctl.get('retry_job'):
                self.state['retry_job']=ctl['retry_job']; self.command(ready,source,'retry',timeout=60)
            else: self.command(ready,source,'resume',timeout=60)
        self.state['control_generation']=ctl.get('generation')
        if ctl.get('stop_active'):
            record = self.state.get('coordinator')
            if alive(record): os.killpg(record['process_group'], signal.SIGTERM)
        if ctl.get('stop'):
            self.state['phase']='STOPPED'; return 'stop'
        return 'pause' if ctl.get('paused') else None

    def run(self):
        lock = (self.root/'receiver.lock').open('a+')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if self.git('remote','get-url','origin').stdout.strip() != REMOTE: raise ValueError('Remote identity changed')
        atomic(self.root/'receiver_process.json',{'pid':os.getpid(),'start_ticks':identity(os.getpid()),'pgid':os.getpgrp(),'started_at':now()})
        ready = read(self.root/'pinned_ready.json'); source = Path(self.state['source_path']) if self.state.get('source_path') else None
        delay=15; consecutive_errors=0
        while True:
            try:
                control = self.controls(ready,source)
                self.state['paused'] = control=='pause'
                if control=='stop': self.save(); self.final_publish(); break
                if control!='pause' and not ready:
                    if time.time()>self.state['deadline_epoch']:
                        self.state['phase']='BLOCKED_READY_TIMEOUT'; self.issue('READY_TIMEOUT','No compatible READY within 12 hours'); self.save(); self.final_publish(); break
                    commit=self.advertised(self.c['shared_ref'])
                    if commit and commit != self.state.get('last_rejected_ready_sha'):
                        try:
                            tree=self.get_source(self.c['shared_ref'],commit); packet=self.get_ready(commit,tree)
                            if packet:
                                candidate,items=packet; files=self.inspect_ready(commit,candidate,items)
                                candidate_source=self.materialize(commit,candidate,files,items)
                                self.state.update(phase='VALIDATING_READY',candidate_ready_sha=commit); self.save()
                                self.launch(candidate,candidate_source)
                                if self.state['phase']=='RUNNING':
                                    ready=candidate; source=candidate_source; atomic(self.root/'pinned_ready.json',ready)
                                    self.state.update(ready_sha=commit,source_path=str(source),protocol_hash=ready['protocol_sha256'])
                                    self.state.pop('last_error',None)
                        except Exception as error:
                            self.issue('READY_INCOMPATIBLE_OR_GATE_FAILED',str(error),commit)
                            self.state.update(last_rejected_ready_sha=commit,last_error=str(error),phase='WAITING_FOR_COMPATIBLE_READY')
                if ready and control!='pause':
                    status=read(self.root/'jobs/queue_status.json',{})
                    if not alive(self.state.get('coordinator')):
                        if status.get('all_terminal') is True:
                            self.state['phase']='LOCAL_TERMINAL_AWAITING_PEER'
                            if self.other_terminal():
                                self.aggregate();self.state['phase']='COMPLETE' if status.get('all_required_validated') and self.state.get('consolidation_status')=='complete' else 'TERMINAL_PARTIAL'
                                self.save();self.final_publish();break
                        attempts=self.state.get('coordinator_restart_attempts',0)
                        if status.get('all_terminal') is True:pass
                        elif attempts<2:
                            self.state['coordinator_restart_attempts']=attempts+1
                            self.launch(ready,source)
                        else:
                            self.state['phase']='BLOCKED_COORDINATOR_EXIT'
                            self.issue('COORDINATOR_EXIT','Coordinator exited without all-terminal validated queue status',self.state['ready_sha'])
                if time.time()-self.last_other>=300:
                    self.collect_other(); self.last_other=time.time()
                    self.aggregate()
                self.save()
                if time.time()-self.last_publish>=self.c['publish_seconds']:
                    self.publish(); self.last_publish=time.time()
                consecutive_errors=0; delay=min(120,delay*2)
            except Exception as error:
                consecutive_errors+=1; self.state.update(last_error=str(error),unsynced=True)
                self.event('receiver_error',error=str(error),consecutive=consecutive_errors)
                self.save(); delay=min(120,30*consecutive_errors)
            time.sleep(delay)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True)
    p.add_argument('action',choices=['run','status','pause','resume','retry','stop','publish'])
    p.add_argument('--job'); p.add_argument('--active',action='store_true'); a=p.parse_args()
    r=Receiver(a.config)
    if a.action=='run': return r.run()
    if a.action=='status':
        print(json.dumps({'receiver':read(r.root/'receiver_status.json'),
                          'queue':read(r.root/'jobs/queue_status.json'),
                          'publication':read(r.root/'publication_receipt.json'),
                          'control':read(r.root/'control.json',{})},indent=2)); return
    if a.action=='publish': return r.publish()
    ctl={'generation':time.time_ns(),'paused':a.action=='pause','stop':a.action=='stop',
         'stop_active':a.action=='stop' and a.active,'retry_job':a.job if a.action=='retry' else None}
    if a.action=='retry' and not a.job: p.error('retry requires --job exact_job_id')
    atomic(r.root/'control.json',ctl); print(json.dumps(ctl))


if __name__=='__main__': main()
