# Finite August P3 overnight execution

Run group: `qf-august-overnight-20261001`. This is new masked training, separate from
the frozen historical August replay. The common code is owned by A5000-labelled
session; each role has its own local queue and result branch. Never execute from a
worktree that is being edited, fetched into by checkout, or used for publication.

The generated manifest contains **247 DAG nodes**: A5000 178, RTX3090 69.
There are 153 unconditional full fits plus at most four conditional fixed-LR fits:
54 shared main fits, 81 S asset fits, 18 new rolling fits, and zero to four extra
canonical U0/U1 seed7/123 fits. Nine R1 nodes, nine selected seed42 nodes and two
fixed seed42 nodes explicitly reuse existing fits. Smoke checks are separate and
never count as scientific completions. Prior nodes cover main/R1/R2/R3. Only TLOB
descendants wait for the bounded optional adapter. A negative predictive result
never blocks anything.

The exact scientific contract is protocol.json. Split manifests use date indices,
not rounded ratio reconstruction. Input histories always retain all38 origins;
loss and checkpoint selection use P3 only. The same original filled full panel is
available to peer models. No P1/P3 endpoint mask affects peer inputs. The loop sums
valid-row cross entropy within each32-day optimizer window, divides accumulated
gradients by that window's valid-row count, then clips and steps. A partial final
window follows the same rule. TF32 and mixed precision are disabled. Float32 is
used for training, and normalized float64 probabilities for reporting NLL.

The existing environment on the implementing host is Python3.9.16, PyTorch1.12.1,
CUDA11.3, NumPy1.22.3, pandas1.5.3, sklearn1.2.2, openpyxl3.1.5. No packages or
drivers were changed. This host's nickname is A5000; its recorded idle devices were
four RTX4090. Names do not determine ownership. Receiver GPU allocation must be
established locally, respecting CUDA_VISIBLE_DEVICES and any reservation. Each
worker receives one physical GPU UUID as CUDA_VISIBLE_DEVICES and uses cuda:0.
CPU threads are capped at2 and DataLoader workers are0. The supervisor never
preempts another project's jobs.

## Receiver initialization

Fetch the advertised `work/qf-overnight-20261001` ref, verify its full SHA and create
a clean immutable execution worktree. The READY record intentionally contains the
prior code SHA and file digests, not its own final commit hash. Inspect the scripts
before using the commands below. Replace explicit uppercase placeholders; commands
do not guess credentials, paths, GPU assignments or a remote.

```sh
export GIT_LFS_SKIP_SMUDGE=1
git -C VERIFIED_BRIDGE fetch --filter=blob:none --no-tags --no-recurse-submodules origin refs/heads/work/qf-overnight-20261001
git -C VERIFIED_BRIDGE rev-parse FETCH_HEAD
git -C VERIFIED_BRIDGE worktree add --detach NEW_EXECUTION_DIRECTORY VERIFIED_READY_SHA
# Create a separate, single-writer result publication worktree from the same SHA.
git -C VERIFIED_BRIDGE worktree add -b results/qf-overnight-20261001-rtx3090 NEW_PUBLICATION_DIRECTORY VERIFIED_READY_SHA
```

Reuse the verified local54 inputs. Prepare a new task-local feature cache using the
common data.prepare function. This is an authorized new training cache; original
workbooks and CSVs are read-only and not rebuilt or overwritten. Verify the80
data-handoff file hashes first using scripts/qf_data_review/verify_handoff.py.

```sh
cd NEW_EXECUTION_DIRECTORY
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 EXISTING_PYTHON -B - <<'PY'
import sys
sys.path.insert(0,'scripts/qf_overnight')
from data import prepare
prepare('VERIFIED_INPUT_ROOT','NEW_TASK_ROOT/prepared')
PY
EXISTING_PYTHON -B scripts/qf_overnight/runtime.py init --root NEW_TASK_ROOT/rtx3090 --owner rtx3090 --cache NEW_TASK_ROOT/prepared --publication NEW_PUBLICATION_DIRECTORY --gpus UUID0,UUID1,UUID2,UUID3
EXISTING_PYTHON -B scripts/qf_overnight/runtime.py start --root NEW_TASK_ROOT/rtx3090
```

Init compares exact prepared feature/factor/label/mask hashes to READY. A numerical
environment discrepancy must be reported through the issue inbox instead of
silently mixing inputs. No installation is implicit. The shared main fits select
LR only from their four seed42 P3 validation-checkpoint NLLs. Per-seed checkpoint
selection remains independent. Every trial's selected score, including unsuccessful
trial status, is exported; no holdout or restricted-score re-selection occurs.

## Controls and persistence

Use the same pinned execution script and existing interpreter:

```sh
EXISTING_PYTHON -B scripts/qf_overnight/runtime.py status --root TASK_ROLE_ROOT
EXISTING_PYTHON -B scripts/qf_overnight/runtime.py pause --root TASK_ROLE_ROOT
EXISTING_PYTHON -B scripts/qf_overnight/runtime.py resume --root TASK_ROLE_ROOT
EXISTING_PYTHON -B scripts/qf_overnight/runtime.py retry --root TASK_ROLE_ROOT --job EXACT_JOB_ID --reason 'Concrete repaired transient failure'
EXISTING_PYTHON -B scripts/qf_overnight/runtime.py stop --root TASK_ROLE_ROOT
EXISTING_PYTHON -B scripts/qf_overnight/runtime.py stop --root TASK_ROLE_ROOT --active
```

Pause stops new claims only; current jobs continue. Stop without --active stops new
claims and lets active jobs finish before workers exit. Stop --active terminates
only recorded live task worker process groups, keeping the last atomically saved
complete-epoch checkpoint. Resume recomputes an interrupted epoch from that
checkpoint's model/optimizer/early-stop/RNG and seeded data-order state; no partial
epoch is mislabeled as an exact continuation. Attempts use separate directories.
Success directories are not retrained or overwritten. Each saved checkpoint is
trusted only as an identified local artifact of this task.

SQLite WAL with BEGIN IMMEDIATE handles claims. PID plus /proc start-time identity
protects live leases even when a heartbeat is old. Dead processes get at most two
infrastructure retries; deterministic errors block affected descendants. OOM gets
one restart with microbatch4 and eight-way accumulation, preserving32 day panels.
No-progress timeout starts at two hours, then uses a generous minimum one hour or
20 observed epoch durations. Disk free space below10GiB pauses new claims; no
historical files are deleted. Loss/validation masks and causal checks are genuine
correctness gates; none depend on model improvement.

The supervisor and workers are detached into recorded process groups with closed
stdin and file logs. They do not depend on this Codex turn or SSH connection. Logs
are under TASK_ROLE_ROOT/logs and runs/JOB/attempt_NNNN/process.log; checkpoints
remain in those local attempt directories. DB/events are task-local; only a JSON
snapshot and append-only JSONL event export enter Git.

## Publication, optional adapters and repairs

The durable publisher exports on phase completion and at least every five minutes,
with a scheduled eight-hour snapshot and a local final report even when peer work
is missing. Active jobs are not stopped at eight hours. Three bounded Git attempts
per cycle tolerate network outages; publication_status.json records unsynced state.
Git success is claimed only after ls-remote matches the pushed SHA. The publisher
stages only results/qf_overnight/20261001/OWNER. Checkpoints, DBs and environments
never enter Git. Trial probability payloads remain local unless selected or needed
for the canonical fixed-LR comparison. Necessary files above45MB use explicit16MiB
lossless parts with byte counts, hashes and reassembly instructions.

Each publisher fetches the other owner's result ref into a task-local peer inbox.
RTX3090 also retrieves selected prediction payloads for consolidated comparisons.
Write a precise common-code issue/reproducer under TASK_ROLE_ROOT/issues/; the
publisher carries it through the role's result branch. The A5000 implementation
session reviews these records. Peer scripts are never automatically executed.

Optional TLOB is added only in new files plus attribution/adapter metadata while
all frozen core files remain byte-identical. Publish compatible READY, create a
new immutable execution worktree and register it using adapter-ready --source
NEW_TLOB_EXECUTION. If bounded integration fails, adapter-blocked --reason records
the exact reason and only its descendants become blocked. Do not mutate running
source. A scientific correction requires a new protocol version and explicit
invalidated job/result list; pause affected claims, publish the correction and
rerun only affected jobs in a new isolated revision queue. Do not relabel old
results with the new protocol hash or silently splice incompatible revisions.

The local owner can finish without the peer. Receiver aggregation marks missing
model/seed support incomplete. Primary UA–UC and UA–UM comparisons use10,000 paired
whole-date moving-block resamples of length5, retaining seeds/assets together and
refit-fold boundaries. Calibration uses15 fixed equal-width confidence bins;
heterogeneity cutpoints use training dates only. All findings remain retrospective,
subject to unknown provider timing/aggregation, historical selection, adaptation
scope and only three optimization seeds. No manuscript edits or trading-profit
claims are part of this run group.
