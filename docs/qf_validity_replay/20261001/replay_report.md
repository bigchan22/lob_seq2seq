# Historical August saved-probability replay

Status: COMPLETE. All nine frozen UA, UC and corrected UM_V2_DEPTH prediction files (seeds 42, 7 and 123) were scored on their full, matched original support. No training, inference, checkpoint loading or model selection was performed in this replay namespace. New P3 training is a separate task and namespace.

## Provenance and checks

- Pinned data handoff: `e85a89fd56584b513be940e6259c637369bd830b`, branch `work/qf-august-data-review-20261001`.
- Recovered code export: `200e5758de0f5cb545322f5b5ab522ad645f9209`; original August source: `feb10a130bb036617229d6c4c9bc4a96df3b9fe6`.
- Published input lineage: `073cd89b556224eeae22413c3c05188a947e990a`. All 54 local input hashes matched `recovery/input_manifest.csv`. No full workbook comparison was repeated.
- The handoff verifier checked 80 task files, all nine compressed/decompressed prediction hashes and the code/input manifests. The independently read 27 original CSVs produced exactly the handoff keys, labels and 41 compared key/endpoint/validity fields for all 505,818 rows; zero per-row mismatches. See `data_mask_checks.json`.
- The 27 assets and 493 common dates cover July 2015–June 2017. “August” is experiment lineage, not a market-calendar filter. Splits retain the exact first 345 / next 49 / last 99 dates.
- Each saved file matched all 151,848 evaluation rows, including validation 50,274 and development holdout 101,574, with no duplicate, extra, missing, mislabelled or wrong-split rows. Explicit YYYYMMDD→ISO conversion follows the producer schema. Endpoint comes from the checked data contract.
- Frozen configuration hashes and validation checkpoint-selection histories matched all selected epochs. UC42 epoch151 is the copied `F_PAIR_5_SHALLOW_INTERACTION_UC_42` fit; it is counted once. UA42 epoch178 and corrected UM42 epoch128 were retained. No restricted score was used to select a file/seed. Per-run historical source SHA/dirty state/execution host remain unknown where the producer cannot establish them.

## Populations and numerical convention

P0 retains all legacy rows. P1 requires original pre-fill finite, strictly positive bid1/ask1 at both endpoints, uncrossed quotes and a same-day ten-minute target; positive locked quotes remain. P2 requires origin≥09:10 and endpoint≤14:40, exactly ten minutes later on the same date (last origin14:30). P3 intersects P1 and P2. Exact-grid joins, within-day forward fill then zero fill were reproduced only for label verification; original values before filling determine P1. No off-grid or cross-day seed is used. Float32 midpoint/return arithmetic followed by scalar-equivalent threshold comparison preserved the original strict ±1bp labels (Down/Flat/Up=0/1/2).

The separate missing-cache diagnostic is 34,600 rows, including 27,366 Flat rows with equal filled mids. P1 excludes 36,427 rows overall, so it is correctly broader than missing-only. Its development holdout excludes 4,418 rows, versus 4,411 missing-only. Presence is not evidence of a provider observation; provider aggregation/filling and release timing remain unknown.

Probabilities in Down/Flat/Up order were checked finite, in [0,1], positive row sum and row-sum error<1e-5 (actual maximum 1.4e-07). Saved probabilities were converted to float64, divided by each row sum, and true-class probabilities clipped below at float64 epsilon. NLL uses negative natural log and equal forecast-row weights. Accuracy, fixed-three-class macro-F1, MCC, three-class sum Brier, 15 fixed-width confidence bins/ECE, and equal-asset mean NLL are also recorded. Three-seed summaries average separately trained model losses/metrics, never probabilities.

## Legacy reconciliation

| Split | UA42 NLL | UC42 NLL | UA−UC |
|---|---:|---:|---:|
| validation | 0.887593050 | 0.903287486 | -0.015694436 |
| development_holdout | 0.908952596 | 0.922755800 | -0.013803204 |

All four supplied anchors agree within 1e-7 absolute (actual maximum error <4.2e-10). This reproduces scoring of historical probabilities, not training.

## Matched three-seed findings

| Split | Population | Rows | UA NLL | UC NLL | UM NLL | UA−UC [95% interval] | UA−UM | UA<UC assets |
|---|---|---:|---:|---:|---:|---|---:|---:|
| validation | P0 | 50274 | 0.887127 | 0.904153 | 0.905952 | -0.017026 [-0.018543, -0.015517] | -0.018826 | 27/27 |
| validation | P1 | 48280 | 0.911237 | 0.920510 | 0.921959 | -0.009273 [-0.011208, -0.007546] | -0.010723 | 25/27 |
| validation | P2 | 43659 | 0.906071 | 0.913049 | 0.915609 | -0.006978 [-0.009240, -0.005290] | -0.009538 | 22/27 |
| validation | P3 | 43050 | 0.907119 | 0.913321 | 0.915869 | -0.006201 [-0.008540, -0.004528] | -0.008750 | 21/27 |
| development_holdout | P0 | 101574 | 0.908244 | 0.922517 | 0.923574 | -0.014272 [-0.016537, -0.011698] | -0.015329 | 27/27 |
| development_holdout | P1 | 97156 | 0.928540 | 0.933820 | 0.934707 | -0.005279 [-0.007773, -0.002521] | -0.006167 | 26/27 |
| development_holdout | P2 | 88209 | 0.926511 | 0.932081 | 0.933365 | -0.005570 [-0.006969, -0.004066] | -0.006854 | 24/27 |
| development_holdout | P3 | 86588 | 0.922666 | 0.926923 | 0.928599 | -0.004257 [-0.005672, -0.002848] | -0.005933 | 24/27 |

For development holdout, UA−UC remains negative after invalid endpoints are removed (P1 −0.005279), after the clock restriction alone (P2 −0.005570), and for their intersection (P3 −0.004257). This is substantially smaller than P0 −0.014272. P3 favours UA in 24/27 assets; validation P3 favours UA in 21/27. Thus the mean direction is broad, with explicit exceptions; every asset and seed is included in `per_asset_differences.csv`, without selecting winning groups.

Absolute losses cannot be compared across these populations as if difficulty/class composition were fixed. The holdout Flat fraction changes from 49.683% (P0) to 47.980% (P1), 49.236% (P2), and 48.916% (P3); Down/Up shares and the discarded clock rows also change. The paired within-population delta answers a different question. Equal-asset differences are reported alongside pooled row weighting, which can differ under nonuniform validity masks.

Population rows, dates, assets, retention, class counts and overlapping exclusions are in `population_counts.csv`. Its reason columns describe the original split before a mask and overlap; they are not an additive decomposition of `excluded_total`. P1/P3 train support has 343 dates with valid rows, while the frozen train date manifest still contains all345 dates.

The exploratory intervals use 10,000 paired non-circular moving blocks of five consecutive trading dates, random seed20261001; blocks are concatenated and truncated to the split date count. Every sampled date retains its complete panel and matched seed set; daily sums/counts reproduce the pooled delta. These are percentile95% intervals conditional on historical selection. Three optimization seeds are not three independent market experiments. The historically inspected holdout is development data, not an untouched confirmation set.

## Additional diagnostics and limits

All origin-hour buckets, current spread and current L5 depth groups are reported in `subgroup_metrics.csv`/`subgroup_differences.csv`. Spread/depth tertiles were fitted only on train-P3 current covariates; invalid values have a separate group. Current covariates use the verified legacy within-day fills and never the future endpoint. Cutpoints and fit counts are retained. Calibration tables contain all15 fixed equal-width bins with counts and observed confidence/accuracy; `calibration_P3_holdout_seed42.svg` is one view. NLL alone is not evidence of improved calibration.

These masks restrict evaluation of historical unmasked-training models; they do not retrain under a revised protocol. Ex-post endpoint validity is not a contemporaneous trading-selection rule. Workbook/cache provenance was inherited from the exact producer handoff; this pass verified original CSV semantics and byte identity, not provider observation or release timing. Historical results cannot identify what P3 training, validation reselection, rolling refits or learned-common-query controls will change. No trading profitability is claimed.

The next authorized experiments are (1) matched P3-trained UA/UC plus shared-query control, (2) frozen-LR retrospective rolling fits with fold-specific train-only transforms, and (3) matched GRU/UM controls when producer outputs become available. They belong to the separate overnight protocol and are not reported as completed here.

## Reproduction

Use an existing CPU Python with the versions in `replay_manifest.json`. From this commit, with `INPUT_ROOT` naming the exact recovered54 input tree:

```sh
python3 -B scripts/qf_data_review/verify_handoff.py --repository .
python3 -B scripts/qf_validity_replay/test_replay_core.py
python3 -B scripts/qf_validity_replay/run_historical.py --repository . --inputs "$INPUT_ROOT" --output /new/output/directory
```

The replay entry point refuses to overwrite a completed replay. `replay_manifest.json` records hashes, the exact scoring implementation, environment versions, matching assertions and bootstrap specification; it excludes itself from its checksum list. Required numerical tables are `metrics.csv`, `paired_differences.csv`, `population_counts.csv` and `per_asset_differences.csv`. Narrow validation includes key/label failures, probability failures, clock/cross-day boundaries, float32 labels and row-weighted bootstrap behavior.
