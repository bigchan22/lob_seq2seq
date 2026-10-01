# August KRX data contract and forecast-row review

Created 2026-10-01T08:38:50.516550+00:00. Evidence statuses below are **verified**, **inferred**, or **unknown**; verification methods are source_inspection, static_code, saved_artifact, and recomputed. This is the accepted August baseline. No September recovery, qf_v2 work, training, checkpoint loading, inference, cache conversion, or model-score selection was performed.

All 54 existing inputs match the pinned manifest by bytes, ordinary Git blob ID, and SHA-256, before and after read-only work. All 27 workbook/cache pairs were fully compared: 688,352 rows and 45 columns per pair schema (4 identifiers, 41 numeric columns). The 505,818-row diagnostic and nine frozen selected prediction files are ready for independent review through Git. Source-mapped code is unchanged (128 files verified).

## Provenance and boundaries

- Verified remote: `git@github.com:bigchan22/lob_seq2seq.git`; task branch `work/qf-august-data-review-20261001` is rooted in the published code-only export `200e5758de0f5cb545322f5b5ab522ad645f9209`. Original committed source is `feb10a130bb036617229d6c4c9bc4a96df3b9fe6`. Input bytes are already published at `073cd89b556224eeae22413c3c05188a947e990a`; no need to publish those 366,098,160 bytes again.
- This session was called A5000, but its single device query reported **4 NVIDIA GeForce RTX 4090, 24,564 MiB each, driver 595.71.05**. This is host provenance only. Work used the existing CPU Python 3.9.16 / NumPy 1.22.3 / pandas 1.5.3 / openpyxl 3.1.5 environment, no installation or GPU invocation.
- Original backup dirty-state caveat remains as described in `recovery/README.md`; this task uses the verified committed export bytes. No existing scientific loader, cache, prediction, checkpoint, checkout, environment, job, or permission was changed. Historical per-run code SHA, dirty state, and execution host remain unknown: current file validation is not proof of a fresh training execution.
- `recovery/README.md` predates the present explicit authorization for Git prediction sharing. Its old artifact-transfer restriction is superseded for the nine scoped files in this task; inherited scientific code remains unchanged.

## Workbook versus CSV, exhaustively verified

Workbook sheets are Sheet1, 45 matching columns; all row keys, their order, missingness, and identifier mappings agree. There are **91 numeric binary-value differences across 84 rows**, all within `atol=rtol=1e-12`, all in BUY/SELL investor-price columns; top quotes are unaffected. There are **0 outside-tolerance discrepancies, 0 missingness mismatches, 0 float32 value mismatches, 0 workbook/cache label mismatches**, and no unmatched/duplicate full file keys. Thus file-value provenance is established for the flags. No unmatched/conflicting forecast exists in this dataset. Minor round-trip precision is an inference consistent with the differences, not a claim of bitwise numeric equality.

All cells of every pair were read; this is not the old three-workbook sample. XLSX access was streaming `read_only=True, data_only=True`; XML inspection found **0 formula elements** and **0 comment/external-link parts**. No recalc or workbook writes occurred. Identifier values use strings; numeric Excel identifiers are serialized as integers without invented zero padding. Excel date tokens remain original integer date tokens for matching. Of 688,352 Excel clock cells, 688,351 decode as time and one as timedelta. In `KR7009150004_t`, 2016-05-11 includes `23:10:00` and `2 days, 2:00:00`; the latter is preserved as elapsed duration, not wrapped to a valid clock. Both are off-grid and never seed legacy fills. Neither input was rewritten.

Files: [pair comparison](workbook_cache_comparison.csv), [every column's comparison and missingness](workbook_cache_columns.csv), [workbook types/sheets/anomalies](workbook_metadata.json), [before hashes](input_verification_before.csv), [after hashes](input_verification_after.csv). The cache converter is statically traced at `lob_forecasting/data/preprocess.py:30-54`; it was not executed. Workbook/cache equality verifies supplied values, **not provider observation or operational availability**. No provider documentation establishes aggregation as instantaneous/end/mean/sum, provider filling, release delay, quantity units, or corporate-action policy. Column names alone do not establish these facts.

## Faithful legacy population

`discrimination_runner.py:43` calls `qf_training.load_data` (`qf_training.py:84-85`), which invokes `bake_tensors(..., False, 1e-4, cpu)`. Sorted CSV stems define the 27-asset universe (`data/preprocess.py:17-28`); there is no recovered economic selection criterion. Dates are their chronological intersection (`data/dataset.py:60-65`): 493 common dates, 2015-07-01–2017-06-30. The union has 494; 2015-07-28 is absent from the intersection. No additional date exclusion was introduced.

The fixed 39-point 09:00–15:20 grid is imposed by the loader, yielding 38 daily origins through 15:10; source files do not universally supply 39 observations. `cli/parallel.py:89-118` left-merges exact grid clocks, then forward-fills each asset-day and zero-fills leading gaps. Off-grid values cannot seed fills. Midpoint is computed after ask/bid float32 conversion, and the float32 return is compared to the Python float threshold: Down=0 for r<-0.0001, Up=2 for r>+0.0001, Flat=1 otherwise (including equality or nonpositive current midpoint). No rounding is added. The diagnostic reproduced this literal loop for every asset-day, then independently matched every saved prediction label. `LOBDataset.__getitem__` has a different intermediate dtype path; this task follows the actual Stage-2 bake path.

Main features exclude investors and contain 13 values: log mid, two relative log top prices, and log1p quantities at five levels, followed by `torch.nan_to_num` (`data/transforms.py:10-37`, `cli/parallel.py:124-125`). No tensors or features were rebuilt here. Whole-date splits below follow floor(0.7N), floor(0.8N); development holdout has been examined and is **not an untouched lockbox**.

| split | days | first_date | last_date | rows |
| --- | --- | --- | --- | --- |
| train | 345 | 2015-07-01 | 2016-11-23 | 353970 |
| validation | 49 | 2016-11-24 | 2017-02-03 | 50274 |
| development_holdout | 99 | 2017-02-06 | 2017-06-30 | 101574 |

## Named masks, retained counts and class proportions

P0 is every legacy row. P1 requires original finite strictly positive bid1/ask1 at both endpoints, no cross, and same-day exact +10min; positive locks are retained. P2 applies only the predeclared origin>=09:10 / endpoint<=14:40 clock (33 origins through 14:30). P3 is their intersection. **P1/P3 are ex-post evaluation populations.** A future label endpoint must never control current peer inputs; the published `peer_origin_available_cache` uses only that peer's origin quote. It remains a cache-quality proxy, not a provider-certified availability flag. No legacy rows are deleted.

Totals: P0 **505,818**; P1 **469,391** (36,427 excluded); P2 **439,263** (66,555 excluded); P3 **432,560** (73,258 excluded). Class percentages are within each retained population. Excluded counts always use the corresponding P0 group as denominator.

| split | population | retained | excluded | down_pct | flat_pct | up_pct |
| --- | --- | --- | --- | --- | --- | --- |
| development_holdout | P0_legacy | 101574 | 0 | 25.4002 | 49.6830 | 24.9168 |
| development_holdout | P1_valid_cache_endpoints | 97156 | 4418 | 26.2423 | 47.9795 | 25.7781 |
| development_holdout | P2_candidate_clock | 88209 | 13365 | 25.5904 | 49.2365 | 25.1732 |
| development_holdout | P3_intersection | 86588 | 14986 | 25.7299 | 48.9156 | 25.3546 |
| train | P0_legacy | 353970 | 0 | 30.5176 | 41.5089 | 27.9736 |
| train | P1_valid_cache_endpoints | 323955 | 30015 | 31.8418 | 38.6193 | 29.5390 |
| train | P2_candidate_clock | 307395 | 46575 | 30.8265 | 40.5280 | 28.6456 |
| train | P3_intersection | 302922 | 51048 | 31.1054 | 39.9641 | 28.9305 |
| validation | P0_legacy | 50274 | 0 | 24.7285 | 51.5296 | 23.7419 |
| validation | P1_valid_cache_endpoints | 48280 | 1994 | 25.5613 | 49.8778 | 24.5609 |
| validation | P2_candidate_clock | 43659 | 6615 | 25.0510 | 50.9196 | 24.0294 |
| validation | P3_intersection | 43050 | 7224 | 25.1940 | 50.6156 | 24.1905 |

All 27 assets and populations are shown below; exact class counts and proportions are in [by asset](population_counts_by_asset.csv). Joint asset × split × population counts are in [asset/split table](population_counts_by_asset_split.csv), with [schedule-era/split](population_counts_by_regime_split.csv) and [origin/split](population_counts_by_origin_split.csv) support.

| asset_id | population | retained | excluded | down_pct | flat_pct | up_pct |
| --- | --- | --- | --- | --- | --- | --- |
| KR7000030007_t | P0_legacy | 18734 | 0 | 21.5384 | 58.0442 | 20.4174 |
| KR7000030007_t | P1_valid_cache_endpoints | 17594 | 1140 | 22.0643 | 56.9626 | 20.9731 |
| KR7000030007_t | P2_candidate_clock | 16269 | 2465 | 21.3965 | 58.3871 | 20.2164 |
| KR7000030007_t | P3_intersection | 16167 | 2567 | 21.5130 | 58.1555 | 20.3315 |
| KR7000100008_t | P0_legacy | 18734 | 0 | 28.0399 | 45.4788 | 26.4813 |
| KR7000100008_t | P1_valid_cache_endpoints | 17368 | 1366 | 29.5025 | 42.9065 | 27.5910 |
| KR7000100008_t | P2_candidate_clock | 16269 | 2465 | 28.8893 | 44.0838 | 27.0269 |
| KR7000100008_t | P3_intersection | 16051 | 2683 | 29.0698 | 43.6421 | 27.2880 |
| KR7000120006_t | P0_legacy | 18734 | 0 | 23.2892 | 55.0710 | 21.6398 |
| KR7000120006_t | P1_valid_cache_endpoints | 17225 | 1509 | 24.4644 | 52.8128 | 22.7228 |
| KR7000120006_t | P2_candidate_clock | 16269 | 2465 | 23.5601 | 54.3365 | 22.1034 |
| KR7000120006_t | P3_intersection | 15946 | 2788 | 23.7552 | 53.9257 | 22.3191 |
| KR7000150003_t | P0_legacy | 18734 | 0 | 25.7553 | 50.7847 | 23.4600 |
| KR7000150003_t | P1_valid_cache_endpoints | 17335 | 1399 | 26.7090 | 48.5896 | 24.7015 |
| KR7000150003_t | P2_candidate_clock | 16269 | 2465 | 25.9020 | 50.0830 | 24.0150 |
| KR7000150003_t | P3_intersection | 15983 | 2751 | 26.1090 | 49.6215 | 24.2695 |
| KR7000270009_t | P0_legacy | 18734 | 0 | 29.7801 | 42.8739 | 27.3460 |
| KR7000270009_t | P1_valid_cache_endpoints | 17613 | 1121 | 30.5343 | 41.0776 | 28.3881 |
| KR7000270009_t | P2_candidate_clock | 16269 | 2465 | 29.6576 | 42.6763 | 27.6661 |
| KR7000270009_t | P3_intersection | 16183 | 2551 | 29.8152 | 42.3778 | 27.8070 |
| KR7000660001_t | P0_legacy | 18734 | 0 | 33.0736 | 36.6660 | 30.2605 |
| KR7000660001_t | P1_valid_cache_endpoints | 17615 | 1119 | 33.5112 | 34.8964 | 31.5924 |
| KR7000660001_t | P2_candidate_clock | 16269 | 2465 | 32.5957 | 36.5173 | 30.8870 |
| KR7000660001_t | P3_intersection | 16183 | 2551 | 32.7690 | 36.1923 | 31.0387 |
| KR7000720003_t | P0_legacy | 18734 | 0 | 36.0734 | 30.7035 | 33.2230 |
| KR7000720003_t | P1_valid_cache_endpoints | 17603 | 1131 | 37.2209 | 28.0804 | 34.6986 |
| KR7000720003_t | P2_candidate_clock | 16269 | 2465 | 36.6034 | 29.2827 | 34.1140 |
| KR7000720003_t | P3_intersection | 16179 | 2555 | 36.8070 | 28.9017 | 34.2914 |
| KR7000810002_t | P0_legacy | 18734 | 0 | 25.4030 | 49.2313 | 25.3656 |
| KR7000810002_t | P1_valid_cache_endpoints | 17447 | 1287 | 26.4974 | 46.8791 | 26.6235 |
| KR7000810002_t | P2_candidate_clock | 16269 | 2465 | 25.7729 | 48.1099 | 26.1172 |
| KR7000810002_t | P3_intersection | 16156 | 2578 | 25.9161 | 47.8212 | 26.2627 |
| KR7001040005_t | P0_legacy | 18734 | 0 | 29.4651 | 43.3063 | 27.2286 |
| KR7001040005_t | P1_valid_cache_endpoints | 17534 | 1200 | 30.6091 | 40.9091 | 28.4818 |
| KR7001040005_t | P2_candidate_clock | 16269 | 2465 | 30.1555 | 42.0616 | 27.7829 |
| KR7001040005_t | P3_intersection | 16157 | 2577 | 30.2841 | 41.7590 | 27.9569 |
| KR7001450006_t | P0_legacy | 18734 | 0 | 29.8601 | 41.4647 | 28.6751 |
| KR7001450006_t | P1_valid_cache_endpoints | 17385 | 1349 | 31.3546 | 38.5735 | 30.0719 |
| KR7001450006_t | P2_candidate_clock | 16269 | 2465 | 30.5305 | 39.7873 | 29.6822 |
| KR7001450006_t | P3_intersection | 16126 | 2608 | 30.7206 | 39.4332 | 29.8462 |
| KR7002380004_t | P0_legacy | 18734 | 0 | 31.9953 | 36.9916 | 31.0131 |
| KR7002380004_t | P1_valid_cache_endpoints | 17392 | 1342 | 33.6132 | 33.7626 | 32.6242 |
| KR7002380004_t | P2_candidate_clock | 16269 | 2465 | 33.1428 | 35.1343 | 31.7229 |
| KR7002380004_t | P3_intersection | 16074 | 2660 | 33.4080 | 34.6149 | 31.9771 |
| KR7002550002_t | P0_legacy | 18734 | 0 | 27.9065 | 45.8311 | 26.2624 |
| KR7002550002_t | P1_valid_cache_endpoints | 16790 | 1944 | 29.6367 | 42.2394 | 28.1239 |
| KR7002550002_t | P2_candidate_clock | 16269 | 2465 | 28.5758 | 44.3543 | 27.0699 |
| KR7002550002_t | P3_intersection | 15574 | 3160 | 29.1511 | 43.1296 | 27.7193 |
| KR7002790004_t | P0_legacy | 18734 | 0 | 25.6913 | 50.4003 | 23.9084 |
| KR7002790004_t | P1_valid_cache_endpoints | 17538 | 1196 | 26.5367 | 48.4776 | 24.9857 |
| KR7002790004_t | P2_candidate_clock | 16269 | 2465 | 25.9389 | 49.8740 | 24.1871 |
| KR7002790004_t | P3_intersection | 16161 | 2573 | 26.0813 | 49.6009 | 24.3178 |
| KR7003550001_t | P0_legacy | 18734 | 0 | 31.6483 | 37.4026 | 30.9491 |
| KR7003550001_t | P1_valid_cache_endpoints | 17536 | 1198 | 33.0349 | 34.6373 | 32.3278 |
| KR7003550001_t | P2_candidate_clock | 16269 | 2465 | 32.4728 | 35.8658 | 31.6614 |
| KR7003550001_t | P3_intersection | 16172 | 2562 | 32.6367 | 35.5244 | 31.8390 |
| KR7004020004_t | P0_legacy | 18734 | 0 | 32.6145 | 38.4381 | 28.9474 |
| KR7004020004_t | P1_valid_cache_endpoints | 17611 | 1123 | 33.6778 | 36.2387 | 30.0835 |
| KR7004020004_t | P2_candidate_clock | 16269 | 2465 | 32.8047 | 37.5807 | 29.6146 |
| KR7004020004_t | P3_intersection | 16181 | 2553 | 32.9769 | 37.2536 | 29.7695 |
| KR7004170007_t | P0_legacy | 18734 | 0 | 25.9635 | 50.2562 | 23.7803 |
| KR7004170007_t | P1_valid_cache_endpoints | 17418 | 1316 | 27.0295 | 48.0480 | 24.9225 |
| KR7004170007_t | P2_candidate_clock | 16269 | 2465 | 26.1110 | 49.8371 | 24.0519 |
| KR7004170007_t | P3_intersection | 16049 | 2685 | 26.3319 | 49.4423 | 24.2258 |
| KR7005300009_t | P0_legacy | 18734 | 0 | 38.7264 | 27.7143 | 33.5593 |
| KR7005300009_t | P1_valid_cache_endpoints | 14870 | 3864 | 44.4317 | 17.1755 | 38.3927 |
| KR7005300009_t | P2_candidate_clock | 16269 | 2465 | 40.7954 | 24.3039 | 34.9007 |
| KR7005300009_t | P3_intersection | 13742 | 4992 | 44.4404 | 17.3774 | 38.1822 |
| KR7005380001_t | P0_legacy | 18734 | 0 | 17.2040 | 66.3660 | 16.4300 |
| KR7005380001_t | P1_valid_cache_endpoints | 17613 | 1121 | 17.3281 | 65.6220 | 17.0499 |
| KR7005380001_t | P2_candidate_clock | 16269 | 2465 | 16.1350 | 67.7485 | 16.1165 |
| KR7005380001_t | P3_intersection | 16183 | 2551 | 16.2207 | 67.5833 | 16.1960 |
| KR7005440003_t | P0_legacy | 18734 | 0 | 25.0027 | 51.6441 | 23.3533 |
| KR7005440003_t | P1_valid_cache_endpoints | 17358 | 1376 | 26.0975 | 49.4873 | 24.4153 |
| KR7005440003_t | P2_candidate_clock | 16269 | 2465 | 25.2812 | 50.8943 | 23.8245 |
| KR7005440003_t | P3_intersection | 16013 | 2721 | 25.4918 | 50.4528 | 24.0555 |
| KR7005490008_t | P0_legacy | 18734 | 0 | 24.8852 | 50.7046 | 24.4102 |
| KR7005490008_t | P1_valid_cache_endpoints | 17612 | 1122 | 25.3691 | 49.1540 | 25.4769 |
| KR7005490008_t | P2_candidate_clock | 16269 | 2465 | 24.2793 | 51.1464 | 24.5743 |
| KR7005490008_t | P3_intersection | 16183 | 2551 | 24.4083 | 50.8929 | 24.6988 |
| KR7005830005_t | P0_legacy | 18734 | 0 | 30.5754 | 39.4203 | 30.0043 |
| KR7005830005_t | P1_valid_cache_endpoints | 17417 | 1317 | 31.9975 | 36.4529 | 31.5496 |
| KR7005830005_t | P2_candidate_clock | 16269 | 2465 | 31.2250 | 37.7466 | 31.0283 |
| KR7005830005_t | P3_intersection | 16126 | 2608 | 31.4027 | 37.3620 | 31.2353 |
| KR7005930003_t | P0_legacy | 18734 | 0 | 36.8741 | 26.4332 | 36.6926 |
| KR7005930003_t | P1_valid_cache_endpoints | 17599 | 1135 | 38.0931 | 23.4559 | 38.4510 |
| KR7005930003_t | P2_candidate_clock | 16269 | 2465 | 37.4577 | 24.8448 | 37.6975 |
| KR7005930003_t | P3_intersection | 16181 | 2553 | 37.6553 | 24.4484 | 37.8963 |
| KR7006400006_t | P0_legacy | 18734 | 0 | 21.7145 | 58.4979 | 19.7876 |
| KR7006400006_t | P1_valid_cache_endpoints | 17609 | 1125 | 22.0853 | 57.3514 | 20.5633 |
| KR7006400006_t | P2_candidate_clock | 16269 | 2465 | 20.9232 | 59.4136 | 19.6632 |
| KR7006400006_t | P3_intersection | 16179 | 2555 | 21.0334 | 59.2064 | 19.7602 |
| KR7006800007_t | P0_legacy | 18734 | 0 | 32.4650 | 39.3669 | 28.1680 |
| KR7006800007_t | P1_valid_cache_endpoints | 17553 | 1181 | 33.4587 | 37.1048 | 29.4366 |
| KR7006800007_t | P2_candidate_clock | 16269 | 2465 | 32.3499 | 38.6994 | 28.9508 |
| KR7006800007_t | P3_intersection | 16124 | 2610 | 32.5540 | 38.2969 | 29.1491 |
| KR7007070006_t | P0_legacy | 18734 | 0 | 33.2177 | 35.9560 | 30.8263 |
| KR7007070006_t | P1_valid_cache_endpoints | 17577 | 1157 | 34.5850 | 33.2935 | 32.1215 |
| KR7007070006_t | P2_candidate_clock | 16269 | 2465 | 34.2307 | 34.5196 | 31.2496 |
| KR7007070006_t | P3_intersection | 16163 | 2571 | 34.4367 | 34.1397 | 31.4236 |
| KR7008770000_t | P0_legacy | 18734 | 0 | 30.9971 | 42.0145 | 26.9884 |
| KR7008770000_t | P1_valid_cache_endpoints | 17588 | 1146 | 31.9138 | 39.8908 | 28.1954 |
| KR7008770000_t | P2_candidate_clock | 16269 | 2465 | 30.9054 | 41.6866 | 27.4080 |
| KR7008770000_t | P3_intersection | 16162 | 2572 | 31.0914 | 41.3439 | 27.5647 |
| KR7009150004_t | P0_legacy | 18734 | 0 | 30.9331 | 40.8882 | 28.1787 |
| KR7009150004_t | P1_valid_cache_endpoints | 17591 | 1143 | 31.6639 | 38.7755 | 29.5606 |
| KR7009150004_t | P2_candidate_clock | 16269 | 2465 | 30.7333 | 40.3836 | 28.8832 |
| KR7009150004_t | P3_intersection | 16162 | 2572 | 30.9306 | 40.0012 | 29.0682 |

## Missingness and price quality

Recomputed historical missing-cache-endpoint result is **34,600 / 505,818 = 6.840405%** (train 28,195; validation 1,994; development holdout 4,411). Current-only=9,365; future-only=20,719; both=4,516. Historical missing means absent/nonfinite original top quotes; there is no infinity-only discrepancy here. Missing endpoint AND equal filled mids AND Flat is **27,366 / 505,818 = 5.410246%**, an overlapping subset, not a second disjoint category. Equal prices alone never identify imputation.

P1 excludes **1,827 additional rows** without historical missingness because of nonpositive or crossed values. The following row counts may overlap and must not be summed as independent causes. Positive locks can coexist with a different endpoint defect; only that defect rejects the row.

| split | rows | historical_missing_endpoint | nonpositive_any_endpoint | crossed_any_endpoint | extra_P1_exclusions_without_missing | locked_positive_any_endpoint | locked_positive_retained_P1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| development_holdout | 101574 | 4411 | 7 | 0 | 7 | 0 | 0 |
| train | 353970 | 28195 | 2014 | 706 | 1820 | 11356 | 5652 |
| validation | 50274 | 1994 | 6 | 4 | 0 | 0 | 0 |

Per-top-quote presence/finite/positive flags, source time, forward-fill/zero-fill/original method, crossing/locking, and full file-value provenance are in every forecast row. [Missingness by asset and split](missingness_by_asset_split.csv) includes per-endpoint and per-quote counts. Origin quote filling: each side has 11,254 forward-filled and 2,627 zero-filled occurrences. Future endpoint: each side has 22,717 forward-filled and 2,518 zero-filled occurrences. A literal zero supplied in the file remains `original`, not zero_fill. These are forecast endpoint occurrences, not unique original-row counts.

## Clock, session regimes and unknown provider semantics

Source/cache rows total 688,352, of which 170,601 are off the imposed legacy grid. Daily counts range 18–55. The 493 common-date population is missing 1,844 asset/grid rows (1,846 if counting all source dates); existing grid rows can separately contain missing quote cells. The input anomaly clocks remain excluded only by the faithful legacy grid. [Daily clock summary](daily_clock_summary.csv) preserves coverage and off-grid counts; [regime summary](clock_regime_summary.csv) is split at 2016-08-01.

Existing local KOFIA 2011 guide PDF pages 26/35 documents 09:00–15:00 regular hours, closing call 14:50–15:00, and first-year-day/exam-day exceptions. Existing KRX 2017 brochure PDF pages 16/30 documents the 30-minute extension on 2016-08-01 and 09:00–15:30 regular hours. Hashes and scope are in [calendar evidence](calendar_evidence.json). Those documents support schedule eras but do not identify every 2015–2017 exceptional session in these files. No halt/auction/exception flags were invented. The historical fixed grid crosses different official schedules; candidate_clock is a provisional restriction, not a certification of continuous trading or executable timing. Local-clock convention is Asia/Seoul; provider timestamp/release semantics remain unknown.

## Frozen historical predictions

The index was frozen at **2026-10-01T08:18:00.284477+00:00**, before population creation and before any restricted scoring; no restricted model scores were computed in this task. Selection follows the existing explicit mapping in `lob_forecasting/experiments/discrimination_finalize.py:16` and the user's pinned paths. All nine originals have 50,274 validation and 101,574 development-holdout rows (151,848 each), despite the filename `test_predictions.csv.gz`. Probabilities map Down/Flat/Up to columns `probability_down/flat/up`, are finite in [0,1], and have maximum row-sum error <=1.4e-7. Stable keys, all labels and both splits agree exactly with this contract; duplicate/missing keys are zero. Prediction endpoints are joined from the verified +10min contract because original predictions store origins only.

| model | seed | run_id | epoch | recorded_val_NLL | key_label_errors |
| --- | --- | --- | --- | --- | --- |
| UA | 42 | F_PAIR_2_DOUBLE_LR_UA_42 | 178 | 0.887593050 | 0 |
| UA | 7 | F_best_UA_7 | 181 | 0.887514265 | 0 |
| UA | 123 | F_best_UA_123 | 196 | 0.886272400 | 0 |
| UC | 42 | F_best_UC_42_stage2 | 151 | 0.903287486 | 0 |
| UC | 7 | F_best_UC_7 | 115 | 0.904503193 | 0 |
| UC | 123 | F_best_UC_123 | 124 | 0.904667401 | 0 |
| UM_V2_DEPTH | 42 | D_UM_V2_DEPTH_42 | 128 | 0.904976156 | 0 |
| UM_V2_DEPTH | 7 | D_selected_7 | 122 | 0.906972538 | 0 |
| UM_V2_DEPTH | 123 | D_selected_123 | 127 | 0.905907917 | 0 |

Table NLL values are **existing training-history entries at the recorded checkpoint**, not newly computed restricted scores. Checkpoint selection was independently replayed from every saved history using validation log loss and strict min_delta=0.0001 (`discrimination_runner.py:53-62`); all nine selected epochs match. Original probability bytes are preserved; compressed and decompressed hashes, configs, histories, manifests and status are in [frozen index](prediction_index.json) and [checks](validation_checks.json). No checkpoint payload is included. Existing flags in predictions that say `unavailable` are preserved; use the diagnostic contract for cache-derived flags, not an invented observation flag.

**UC metadata conflict:** `F_best_UC_42_stage2` epoch151 equals `F_PAIR_5_SHALLOW_INTERACTION_UC_42` in decompressed predictions/config/history/manifest and is counted once. The legacy selection YAML and Stage-2 decision instead name PAIR_0_BASELINE. We preserve both records and explicitly use the pinned finalizer path; this does not resolve the historical prose inconsistency or justify a new winner. Some PAIR names share identical numeric config, so config equality alone is not artifact identity. Per-run provenance remains separate from configured architecture. The old pilot family is not substituted.

UM_V2_DEPTH is verified in all selected manifests (epochs128/122/127). The corrected implementation uses leave-target-out contemporaneous/backward factors, log1p total five-level depth, train-only .001/.999 clipping and training mean/std (`discrimination_factors.py:7-45`). It is not the old unscaled UM. Seeds7/123 are matching selected Stage-2 artifacts, not new runs or an ensemble. This report makes no September-anchor or untouched-test claim.

## Checks, readiness and receiver action

Executed: exact input hashes before/after; the inspected recovery helper's read-only verify-only check of all 54 paths against the published commit tree/blob identities; exhaustive workbook/cache keys/schema/values/missingness; all asset-day literal float32 label equivalence; unique keys and split boundaries; synthetic strict-threshold/nonpositive/fill/off-grid/cross/lock/clock/future-peer checks; nine saved probability/key/label/split checks; nine history-based epoch replays; original code-manifest hash checks; syntax checks of task helpers. The initial optional `pdftotext` attempt found the executable absent; existing `pypdf` successfully read the local references. Missing-path search probes were corrected by reading the actual repository paths. An optional report-wording probe failed because it required a literal word rather than testing data; it is not part of scientific validation. No environment changes were needed.

Data-ready means the diagnostic contract and masks are verified, not that provider semantics or a new scientific experiment are approved. Predictions-ready means these nine frozen files support independent rescoring, not that this session reproduced model training. No packages are missing for this CPU task. Pending: provider semantics; exact dated exception/halt calendar if needed later; UC selection-description conflict; historical per-run execution identity. Legacy Stage-2 hardcoded paths remain unchanged and would require separate runtime work before future training.

The receiving RTX3090 session should verify the advertised task SHA, fetch this exact branch into a separate worktree, read [handoff.json](handoff.json), validate its file hashes, then join probabilities to masks by the explicit key mapping in the contract. Report P0 first and P1/P2/P3 separately, with validation and development_holdout separate, per seed; preserve the frozen selections and do not construct a probability ensemble by default. Any rescoring must document natural log, normalization/clipping and row weighting. The legacy trainer normalized probabilities and called sklearn log_loss (`qf_training.py:30-31`, `evaluation/qf_metrics.py:10-12`); the old finalizer used its own direct loss helper. Library-dependent clipping defaults are not assumed identical. No scores were recomputed here.

Inputs are reusable from the published input SHA and `recovery/input_manifest.csv`. Git transfers committed mask/prediction artifacts in this task; it does not automatically transfer any other untracked files. This handoff is a candidate comparison baseline and does not authorize overwriting receiving-server work. No unavailable-server action is required.

Run instructions and precise file formats are in [schema.md](schema.md), [data_contract.json](data_contract.json), and [receiver.md](receiver.md). The final published SHA is delivered outside these committed files, avoiding self-reference.
