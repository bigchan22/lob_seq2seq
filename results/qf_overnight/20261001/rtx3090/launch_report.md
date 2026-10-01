# RTX3090 finite queue launch receipt

The producer bridge and four allocated RTX3090 workers are running detached. All canonical input hashes and common real-data gates passed. This is a launch/progress receipt, not a claim that the finite experiment queue has finished.

Scientific protocol hash: `bd1488a2d411295d7a2118c50ddcc0b158786123613af1676062e2270d1637d1`.
Pinned execution/READY commit: `e2212c40566e21c680a98186ebd8e984aa80d4fd`.
Receiver wire-protocol byte hash: `3cfd389bbd05ec9201a41c0892e29bf8199198d1b216b9a8061370d4240aa583`.
Data handoff: `e85a89fd56584b513be940e6259c637369bd830b`.
Exact prepared cache payload commit: `19b9b56ff865cec3a24f500727015b33a201e93b`.
Historical replay commit: `dee398fb13078937833329a49f0665eb3c459b47`.

The initial675a candidate and the accepted e221 revision have identical models, data transformations, trainer, tuning plan and metric functions. The reviewed update fixes exact saved-label value checks across integer storage widths and adds the receiver transport. Its changed executable digest is recorded; it is not relabelled as identical source. See source_compatibility_review.json. Later Git cache/optional-adapter publications supply exact needed bytes without silently changing the executing e221 source.

All54 source hashes and505818 source-cache label/mask rows were independently checked during the published historical replay. The nine frozen historical files reproduced seed42 anchors within4.2e-10. The main holdout UA−UC three-seed mean differences are P0−0.014272, P1−0.005279, P2−0.005570 and P3−0.004257. P3 is an ex-post restricted replay of historical unmasked training, separate from this new P3-trained queue.

Native DAG:247 global nodes;153 unconditional fits plus up to4 fixed-LR conditional fits. RTX3090 owns69 nodes containing30 new fits (18 main UA/UC/SHARED_QUERY plus12 R2/R3 UA/UC fits),6 R1 reuse nodes and3 intervention jobs with two conditions each. Historical replay was adopted as a reference to its fully verified existing commit. It was not recomputed or counted as new training. All LR trials and selected fits remain separately identified.

Four actual main UA seed42 LR trials started, produced advancing heartbeats/logs and ~9.8MB atomic resume checkpoints. One trusted task checkpoint was independently loaded on CPU and checked for matching run identity, model/optimizer/early-stop/history, Python/NumPy/Torch CPU/CUDA RNG, next-epoch order and effective32-day window. Only metadata/hashes are published, never checkpoints. Epoch times observed initially were about1.1–1.5 seconds; the common worker uses a generous minimum1-hour /20-epoch no-progress threshold, retaining last checkpoint.

A task-local venv supplies NumPy1.22.3, pandas1.5.3 and sklearn1.2.2; the existing torch2.7.1+cu118 GPU installation and original environments are unchanged. A separate CPU-only Torch1.12.1 preparation also reproduced all5 frozen arrays exactly; no GPU package was installed. Exact x/raw-factor bytes were separately retrieved from the published Git cache payload with compressed/decompressed hashes. Original mismatched task-local caches are preserved. Training Torch differs from the producer runtime and is recorded per fit; numerical training trajectories are not claimed to be bitwise cross-host reproductions.

The outer receiver is the single Git writer on this result branch; the A5000 bridge owns local atomic queue claims, GPU/CPU workers, checkpoints, selection and exports. Both execute immutable source copies. The receiver checks the peer branch every5minutes, uses bulk scoped metadata fetching, verifies complete support and scientific hashes, and updates matched three-seed/date-block analyses. Missing/blocked rows are explicit. The independently checked artifact analysis reads selected saved probabilities only. Historical output and new P3 output namespaces remain separate.

Use task-local `control.py status`, `pause`, `resume`, `retry --job EXACT_JOB_ID`, `stop`, or `stop --active`. Pause and default stop stop new claims; default stop also stops the outer receiver while current workers finish. Resume restarts the receiver if needed. Explicit stop --active additionally terminates only the recorded task coordinator process group and its workers; resume starts from preserved complete-epoch checkpoints. Exact absolute paths and current PIDs are in the local receipt/final user message. No unrelated job is signalled.

Publication and analysis continue without an LLM/client connection. Queue_status.json/events.jsonl are snapshots, never a live SQLite DB. Eight-hour progress snapshots do not stop active jobs. Finalization depends on terminal local/peer jobs and retains explicit partial/blocked counts. No manuscript, main branch, prior scientific checkout, original data or historical predictions was modified. Limits include unknown provider aggregation/release timing, retrospective historical selection and only three optimization seeds; no trading profitability or causality claim.
