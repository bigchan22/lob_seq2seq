# RTX3090 receiver instructions

Read `docs/qf_data_review/20261001/handoff.json` at `refs/heads/work/qf-august-data-review-20261001` after verifying the full advertised SHA supplied in the publication receipt. Preserve your own newer/dirty work; use a new isolated worktree. The task's parent is the published August code-only root `200e5758de0f5cb545322f5b5ab522ad645f9209`. No old scientific history is needed for masks/predictions.

```sh
export GIT_LFS_SKIP_SMUDGE=1
git -C YOUR_VERIFIED_BRIDGE fetch --filter=blob:none --no-tags --no-recurse-submodules origin refs/heads/work/qf-august-data-review-20261001
git -C YOUR_VERIFIED_BRIDGE rev-parse FETCH_HEAD
# Check against DATA_HANDOFF_SHA before the next command.
git -C YOUR_VERIFIED_BRIDGE worktree add --detach NEW_DATA_REVIEW_DIRECTORY FETCH_HEAD
```

All new task payloads total about 32 MB; no checkpoint payload is present. Historical code-only ancestry is small. No blanket historical checkout or unfiltered clone is needed. If this bridge's remote is not the verified `git@github.com:bigchan22/lob_seq2seq.git`, stop and resolve identity before fetching. Do not change an existing remote blindly.

First verify the committed packet without any writes or third-party packages:

```sh
python3 -B scripts/qf_data_review/verify_handoff.py --repository .
```

The receiver may then run the inspected bounded CPU validator with an existing Python >=3.9 containing compatible numpy/pandas/openpyxl (recorded versions are in handoff.json; NumPy scalar-promotion behavior is part of the legacy arithmetic and must not be silently changed):

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B scripts/qf_data_review/validate_review.py --repository .
```

This writes only the task validation JSON, deterministically from supplied artifacts, and performs no model scoring, inference, cache conversion or training. Avoid overwriting independently modified task results: run from a fresh task worktree. Source workbooks are not required for rescoring already-supplied masks/predictions.

To independently reproduce the full audit, use a fresh isolated copy of the exported code and copy/import the frozen replay inputs/index first. `build_data_review.py --inputs EXISTING_VERIFIED_INPUT_ROOT --repository NEW_AUDIT_ROOT` compares every workbook and creates new diagnostic masks. It refuses to overwrite forecast_rows.csv.gz. `freeze_predictions.py` is only for a destination without replay_inputs; it requires an existing Stage-2 artifact root, and is unnecessary on the receiver because all needed artifacts are already committed here. Then run validate_review.py and finalize_packet.py. Never execute a legacy queue/training launcher for this verification.

Existing inputs: inspect `recovery/materialize_inputs.py` before use. Its `--verify-only` mode checks a matching local input root; it writes nothing. If missing, use the documented blob-filtered fetch of `073cd89b556224eeae22413c3c05188a947e990a` and restore only its 54 manifest-listed files to a NEW destination. Those files are already-published ordinary Git blobs. No cache rebuilding or checkpoint downloading is needed.

Join keys: prediction asset -> asset_id (preserve _t); prediction date YYYYMMDD -> date YYYY-MM-DD; timestamp -> origin_time. Check a one-to-one outer join before filtering; endpoint_time is same-date +10min from the contract. Each run has 151,848 legacy evaluation rows. Select the original split first, then the named population; report each population's support separately. Predictions are saved probabilities, not logits. Class order is [Down,Flat,Up]=[0,1,2]. No seed ensembling or new checkpoint selection is authorized by this handoff.

P1/P3 eligibility uses the future endpoint and must never become an origin-time peer-input mask. Use only peer_origin_available_cache if inspecting origin cache quality, while retaining unknown provider release timing. Provider aggregation/filling semantics and the precise exceptional-day calendar remain unresolved.
