# August validity replay — partial readiness report

Status: **WAITING_FOR_DATA_HANDOFF**. Updated 2026-10-01T08:25:01.026472+00:00. No production mask comparison, probability rescore, or uncertainty interval has been computed. No training/inference/checkpoint load ran.

## Pinned provenance and Git availability

Established remote: `git@github.com:bigchan22/lob_seq2seq.git`. Required producer ref: `refs/heads/work/qf-august-data-review-20261001`. It was not advertised on the initial query or the final query at 2026-10-01T08:21:28.328912+00:00; there were no conflicting candidate task refs. Consequently **DATA_HANDOFF_SHA is unavailable** and the expected `docs/qf_data_review/20261001/handoff.json`, row/mask contract, frozen artifact index and Git replay_inputs are unavailable.

The progress branch is `analysis/qf-august-validity-replay-20261001-waiting-081733`, based on the already published code-only August export `200e5758de0f5cb545322f5b5ab522ad645f9209`. It is explicitly **not** a producer-handoff-derived analysis branch. The canonical `analysis/qf-august-validity-replay-20261001` name is reserved for a future branch based on the exact producer commit. The export maps to original August source `feb10a130bb036617229d6c4c9bc4a96df3b9fe6`; existing inputs come from published commit `073cd89b556224eeae22413c3c05188a947e990a`.

The 27 workbook and 27 CSV files remain present at verified recorded sizes (366,098,160 bytes); previous streaming file hashes are retained. No input was copied, changed or converted. Actual cache rows were not read to manufacture a missing producer mask. See `input_availability.json` and the existing input manifest referenced there.

## Specific missing artifacts

The known local August artifact roots listed in `local_prediction_inventory.json` do not contain saved predictions. Prior Git metadata identifies these required primary files, but **metadata hashes are not the payloads**:

| Model | Selected artifact | Epoch | Saved probability SHA-256 | Bytes |
| --- | --- | ---: | --- | ---: |
| UA, seed 42 | `F_PAIR_2_DOUBLE_LR_UA_42/test_predictions.csv.gz` | 178 | `dc0fcedcd0926b50ba6ae59755bbcb1294fa036fc00382771ea8220ac8ade1d9` | 3,116,576 |
| UC, seed 42 | `F_best_UC_42_stage2/test_predictions.csv.gz` | 151 | `daf1fa7a74a03343021ea37893391e530331437e991c7daf0f53e6131bc46398` | 3,114,888 |

UC is a copy of `F_PAIR_5_SHALLOW_INTERACTION_UC_42` and must count as one scientific fit. Associated resolved configs and the selected-configurations record are also absent; their known hashes and bytes are in `artifact_requirements.json`. The IDs/epochs above are task-declared anchors pending verification against the frozen producer index, not newly selected models. Corrected UM_V2_DEPTH and matching seed-7/123 selected artifacts are not identified and are not included. No missing file was regenerated.

Publish the producer handoff, contract/digests, frozen artifact index and minimal selected replay_inputs through the required Git ref. No old-server action or non-Git transfer is required. The concrete request is also recorded in `issues/001_missing_data_handoff.json` for the producer to fetch. This issue asserts an availability blocker, not a mask/data inconsistency.

## Data contract and four-population result status

Task-declared expectations, **not independently reproduced here**: 27 assets × 493 common dates × 38 origins = 505,818 labels; train 353,970, validation 50,274 and development holdout 101,574. Down/Flat/Up = 0/1/2, strict +/-0.0001 thresholds on the original float32 return calculation; equality Flat. The missing-only diagnostic references 34,600 labels, including 27,366 Flat/equal-filled-mid labels; this must be checked separately and must not be substituted for P1's positive/finite/uncrossed requirements.

| Population | Validation UA−UC NLL | Development holdout UA−UC NLL | Observed coverage / status |
| --- | --- | --- | --- |
| P0 legacy | — | — | WAITING_FOR_DATA_HANDOFF and saved probabilities |
| P1 valid_cache_endpoints | — | — | Original-cache endpoint flags unverified |
| P2 candidate_clock | — | — | Canonical origin/endpoint contract unverified |
| P3 P1 ∩ P2 | — | — | Neither actual component mask verified |

P1 requires finite strictly positive bid1/ask1 at both original cache endpoints, uncrossed quotes and a same-day ten-minute target. P2 requires origin >=09:10, endpoint <=14:40, exactly ten minutes on the same date (last origin14:30). P3 is their intersection. These are predeclared definitions; this partial task produced no substitute row masks. The four required result CSVs contain **headers only**, not zero counts or fabricated scores.

## Legacy reconciliation target and scoring specification

Reference scores supplied in the task, **not newly measured**:

| Split | Expected rows | UA NLL | UC NLL | UA−UC |
| --- | ---: | ---: | ---: | ---: |
| Validation | 50,274 | 0.887593050 | 0.903287486 | -0.015694436 |
| Development holdout | 101,574 | 0.908952596 | 0.922755800 | -0.013803204 |

Predeclared reconciliation tolerance: absolute NLL/delta deviation <=5e-9 against these nine-decimal references for the exact selected files. Material discrepancies must be resolved before interpreting restricted scores. Frozen file/config/epoch identities and exact canonical `(asset_id,date,origin_time)` support/labels must match first, with endpoint/split assigned from the producer contract. No inner join is permitted to discard missing forecasts. The file name `test_predictions.csv.gz` does not identify its split; use its actual split column.

The prepared scorer verifies finite probabilities in [0,1], positive sums and maximum sum deviation <=1e-5 (small ~1e-7 serialization errors are expected; larger errors fail for investigation). It converts to float64, normalizes each row, selects the true class, clips below float64 epsilon and averages negative natural logarithms with equal row weights. Accuracy, fixed-three-class macro-F1 and MCC are secondary. Production per-asset/equal-asset summaries and identical-seed-set loss averaging remain blocked on artifacts; no probability ensemble is planned.

The exploratory interval is predeclared as 10,000 non-circular moving blocks of five consecutive trading dates, fixed seed20261001, whole asset/model/seed panel per date, concatenation truncated to the original number of split dates, pooled row-weighted paired loss difference and percentile95% bounds. Daily sums/counts make it CPU-efficient. Synthetic tests verified weighted aggregation and deterministic resampling; **no market-data bootstrap was run**.

## Completed CPU work and remaining integration

Recorded this session once: host `gpusystem`, four RTX3090 GPUs (24,576 MiB each, driver515.57); no GPU compute. Existing CPU environment: Python3.9.13, NumPy1.21.5, pandas1.4.4, scikit-learn1.0.2. No packages installed and no PyTorch import needed.

Six targeted synthetic tests passed: normalization/clipping, rejection of invalid probabilities, exact row/label matching and order independence, raw endpoint/clock boundaries, float32 labels with no cross-day fill, and paired row-weighted date-block resampling. `preflight.py` correctly exits3 with WAITING_FOR_DATA_HANDOFF without a pinned SHA. See `validation_checks.json`; reproducible commands are in `scripts/qf_validity_replay/README.md`.

The CPU primitives and preflight are preparation, not an end-to-end replay adapter. Once the actual handoff arrives, inspect its serialization/schema, verify all referenced checksums and lineage, implement/review the explicit adapter, reconstruct essential original-cache flags/labels independently and compare each row or the producer-defined deterministic digests. Freeze eligible prediction artifacts before computing restricted scores. This work must occur before any of the blank result tables can be filled.

## Interpretation limits and next action

It is currently unknown whether the UA−UC difference survives valid-endpoint or clock restrictions, or is broad across assets. No result is inferred from historical anchors or synthetic tests. Masks use future endpoint validity ex post and are not executable trading-selection rules. Supplied valid cache/workbook values do not prove provider observation, aggregation or release timing. Restricted rescoring cannot establish performance after retraining under a revised population. Any eventual block interval is conditional on historical selection and retrospective, not an untouched confirmation; optimization seeds are not independent market samples.

Next action: publish the exact producer ref with the missing contract/index/probability files, then pin its SHA and finish the matched-population analysis. No new experiment recommendations are made before that evidence is available. No rolling evaluation, GRU/shared-query training, tuning or inference was launched.
