# August P3 overnight launch receipt

Snapshot UTC: 2026-10-01T16:04:50.063710+00:00. Role: A5000-labelled owner; actual device inventory recorded at launch was four RTX4090 GPUs. No hardware recovery was repeated.

RUN_GROUP=qf-august-overnight-20261001
PROTOCOL_HASH=bd1488a2d411295d7a2118c50ddcc0b158786123613af1676062e2270d1637d1
READY_REF=refs/heads/work/qf-overnight-20261001
READY_SHA=19b9b56ff865cec3a24f500727015b33a201e93b
ACTIVE_TLOB_CODE_SHA=08fcbdac350a00b7ca587d0c95c803766c01e9bf
QUEUED_CODE_SHA=19b9b56ff865cec3a24f500727015b33a201e93b
RESULT_BRANCH=results/qf-overnight-20261001-a5000
LAST_VERIFIED_RESULT_SHA=07aae73a526a78d34e23bb861f3aef0dda8033e7

Current DAG states: {"PENDING": 25, "RUNNING": 4, "SUCCEEDED": 149}. Full training fits completed:113 (S81, UM_V2_DEPTH12, U1 eight, GRU_U1 six, UX six). Four TLOB LR searches are active; remaining TLOB confirmations/U0/canonical U0 and analysis nodes are queued. Do not interpret149 successful DAG nodes as149 completed training fits. The global bound is153–157 distinct full fits, with static ownership and no extra sweeps. RTX3090 owns30 of those fits. Their last published launch status was still awaiting a reviewed source/cache repin; this host does not claim those jobs have started.

READY_CORE=true; TLOB_ADAPTER_READY=true. Nine core model smokes and the official TLOB adapter passed. Source labels/masks, causal histories, masked accumulation including partial windows, own-only invariance, train-only factors and queue restart gates passed.113 full fits have validated prediction outputs. The CSV int8/int64 validation defect was corrected without changing training or scores; affected analysis nodes were retried and original attempts retained.192 selected metric rows across32 frozen supports were independently checked against the original handoff keys; daily loss sums reproduce NLL within1e-12.

Core source changes since675a are operational and explicitly reviewed in docs/qf_overnight/20261001/compatible_core_revisions.json. Original successful fits remain valid. Every run records its actual source commit/digest; current source files were never mutated under live jobs. The canonical scientific protocol hash is distinct from receiver_protocol.json's byte hash3cfd389bbd05ec9201a41c0892e29bf8199198d1b216b9a8061370d4240aa583.

Exact frozen x.npy and raw_factors.npy were published as gzip files through Git to resolve Torch-version feature-byte differences on RTX3090. Both decompressed SHA256 values match the original READY. Source workbooks/caches were not overwritten. Read prepared_cache_manifest.json; keep mismatched receiver-prepared files as evidence.

Supervisor PID=2180189. Worker/publisher records: {"cpu": {"pid": 2180193, "start": "1102474508", "pgid": 2180193, "alive": true}, "gpu0": {"pid": 2180194, "start": "1102474508", "pgid": 2180194, "alive": true}, "gpu1": {"pid": 2180195, "start": "1102474508", "pgid": 2180195, "alive": true}, "gpu2": {"pid": 2180196, "start": "1102474508", "pgid": 2180196, "alive": true}, "gpu3": {"pid": 2180197, "start": "1102474508", "pgid": 2180197, "alive": true}, "publisher": {"pid": 2180198, "start": "1102474509", "pgid": 2180198, "alive": true}}. Each has a recorded process start tick and its own task process group. Four live TLOB jobs have fresh heartbeats, epoch logs, and resumable full-state last_checkpoint.pt files. Closing this client does not stop them.

Local logs: ${OVERNIGHT_ROOT}/a5000/logs/
Live job logs/checkpoints: ${OVERNIGHT_ROOT}/a5000/runs/<job>/attempt_NNNN/
Published report worktree: ${OVERNIGHT_ROOT}/publication-a5000/results/qf_overnight/20261001/a5000/progress_report.md
Git report path: results/qf_overnight/20261001/a5000/progress_report.md

```bash
# Status
${TASK_PYTHON} -B ${OVERNIGHT_ROOT}/execution-aggregate-fix/scripts/qf_overnight/runtime.py status --root ${OVERNIGHT_ROOT}/a5000
# Pause new claims; active jobs continue
${TASK_PYTHON} -B ${OVERNIGHT_ROOT}/execution-aggregate-fix/scripts/qf_overnight/runtime.py pause --root ${OVERNIGHT_ROOT}/a5000
# Resume claims / restart task supervisor if needed
${TASK_PYTHON} -B ${OVERNIGHT_ROOT}/execution-aggregate-fix/scripts/qf_overnight/runtime.py resume --root ${OVERNIGHT_ROOT}/a5000
# Reviewed retry of one failed/blocked job, bounded by the two-retry budget
${TASK_PYTHON} -B ${OVERNIGHT_ROOT}/execution-aggregate-fix/scripts/qf_overnight/runtime.py retry --root ${OVERNIGHT_ROOT}/a5000 --job EXACT_JOB_ID --reason 'reviewed defect or transient issue resolved'
# Stop new claims; active jobs finish
${TASK_PYTHON} -B ${OVERNIGHT_ROOT}/execution-aggregate-fix/scripts/qf_overnight/runtime.py stop --root ${OVERNIGHT_ROOT}/a5000
# Stop only this task's active worker groups; preserve completed-epoch checkpoints
${TASK_PYTHON} -B ${OVERNIGHT_ROOT}/execution-aggregate-fix/scripts/qf_overnight/runtime.py stop --active --root ${OVERNIGHT_ROOT}/a5000
```

A task-local checkpoint is the epoch-boundary resume authority. Successful attempts are immutable; no queue database or checkpoints are sent through Git. The single publisher batches completions/status every five minutes, retries short network failures, fetches peer status/issues, and writes progress_8h.md at approximately23:38 UTC plus final_report.md at local queue exhaustion. These scripts continue without another LLM turn. A missing peer final report does not prevent local completion.

Provider aggregation/release timing remains unknown. These are retrospective development/rolling evaluations, with P3 endpoint quality defining an ex-post evaluation/training-loss population, never a current peer-availability or executable trading rule. No manuscript changes or claims of profitable trading were made.

## Peer update after the launch snapshot

RTX3090 result commit631593eeb8c30998aaca51521386f97c0e49b85e reports RUNNING at2026-10-01T16:07:02 UTC. Coordinator PID3832630 started from common codee2212c40566e21c680a98186ebd8e984aa80d4fd with four allocated RTX3090 UUIDs. Canonical scientific protocol is unchanged; receiver transport protocol is the separately documented byte hash. Its first status does not yet contain materialized per-job counts, so this confirms coordinator launch, not completion or GPU training. The earlier BLOCKED status above is superseded by this published update.
