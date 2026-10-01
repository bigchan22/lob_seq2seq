# CPU validity replay preparation

Status: WAITING_FOR_DATA_HANDOFF. These are reviewed CPU primitives and a pinned-handoff preflight, not a completed producer-schema adapter or a production replay pipeline. Actual masks, labels, key digests and scores have not been computed. No model, checkpoint, tensor cache, project launcher, or training code is imported.

From this worktree, using the existing CPU environment:

```bash
CPU_PYTHON="${CPU_PYTHON:-$HOME/anaconda3/envs/mpnn/bin/python}"
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$CPU_PYTHON" -B -m unittest discover -s scripts/qf_validity_replay -p test_replay_core.py -v
"$CPU_PYTHON" -I -B scripts/qf_validity_replay/preflight.py --repository "$PWD"
```

The second command deliberately exits 3 with WAITING_FOR_DATA_HANDOFF when no pinned producer SHA is supplied. It does not fetch, resolve a moving tip, or score data. After the handoff is fetched and inspected, pass `--producer-sha FULL_40_CHARACTER_SHA` and (if needed) `--handoff-path REPOSITORY_RELATIVE_PATH`. A present handoff exits 4 with its exact content checksum and top-level schema fields, requiring explicit schema/manifest review before production replay. These nonzero statuses must not be interpreted as successful scoring.

Required producer ref: `refs/heads/work/qf-august-data-review-20261001`; expected file `docs/qf_data_review/20261001/handoff.json`. Use its exact pinned commit, then implement/review the adapter against the actual producer format. Do not invent a producer schema from this receiver's helper signatures. Fetch referenced blobs selectively; input CSVs are already local. See `docs/qf_validity_replay/20261001/artifact_requirements.json` for exact known saved prediction hashes.

`replay_core.py` provides:

- Exact-string canonical key and exact-label matching; missing, extra or duplicate forecasts raise errors before scoring. An explicit reviewed adapter must map the producer/prediction columns without silent key coercion.
- Finite [0,1] Down/Flat/Up probabilities, positive sums, and a predeclared maximum row-sum deviation of 1e-5. Larger deviations fail. NLL uses float64 row normalization, float64 epsilon clipping, natural logs and equal forecast-row weights.
- Independent per-day forward/zero filling and float32 midpoint-return labels, for comparison with the producer after its date/grid contract is verified. No production cache rebuilding.
- Raw-before-fill endpoint validity and clock formulas implementing the user's predeclared P0/P1/P2/P3 definitions; only synthetic examples have been checked. The missing-only 34,600 diagnostic must be separately reconciled using the producer's documented definition.
- A non-circular moving-date-block bootstrap: 10,000 replicates, length five, fixed seed 20261001, concatenate blocks and truncate to the original date count. Supply daily paired loss sums and row counts, preserving zero-row dates. For multiple seeds, first average separately trained models' losses on the same rows and same seed set; never average probabilities. This helper alone does not establish correct real-data support.

The local digest helper documents its own CSV serialization. It is not a substitute for independently reproducing the producer's documented digest serialization. Count equality alone is insufficient.

When the pinned handoff is available, create the requested unsuffixed analysis branch from that producer commit and copy this analysis-only work into it; preserve the current progress branch. Recheck any new AGENTS.md and entrypoint side effects. Do not rebase/reset/merge histories or alter the producer branch.
