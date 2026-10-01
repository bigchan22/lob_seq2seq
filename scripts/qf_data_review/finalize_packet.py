"""Assemble the review contract/report/handoff from completed CPU checks. No scores."""
import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from build_data_review import GRID, POPS, sha

DOCS = 'docs/qf_data_review/20261001'
ARTIFACTS = 'artifacts/qf_data_review/20261001'
SOURCE_SHA = 'feb10a130bb036617229d6c4c9bc4a96df3b9fe6'
EXPORT_SHA = '200e5758de0f5cb545322f5b5ab522ad645f9209'
INPUT_SHA = '073cd89b556224eeae22413c3c05188a947e990a'
REMOTE = 'git@github.com:bigchan22/lob_seq2seq.git'
BRANCH = 'work/qf-august-data-review-20261001'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def table(records, columns):
    return '\n'.join(['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |'] +
                     ['| ' + ' | '.join(str(row.get(c, '')) for c in columns) + ' |' for row in records])


def describe_column(name):
    exact = {
        'asset_id': 'String equal to input filename stem, including _t; no issuer-name inference.',
        'date': 'Common date serialized YYYY-MM-DD; Asia/Seoul local-clock convention.',
        'origin_time': 'Forecast origin HH:MM:SS; fixed grid 09:00:00 through 15:10:00.',
        'endpoint_time': 'Same-date target endpoint HH:MM:SS, exactly origin plus ten minutes.',
        'split': 'train, validation, or development_holdout; whole common dates, chronological.',
        'legacy_label': '0=Down, 1=Flat, 2=Up using the original float32 +/-0.0001 rule.',
        'no_cross_day_target': '1 iff origin/endpoint are on the same date; all rows are 1.',
        'exact_ten_minute_target': '1 iff endpoint is exactly origin plus ten minutes; all rows are 1.',
        'scheduled_session_regime': 'pre_20160801_extension or from_20160801_extension; documented schedule era, not a per-day session certification.',
        'exceptional_session_status': 'Always unresolved; no inferred halt, holiday, auction, or exception-day flag.',
        'origin_history_file_conflict': 'Any workbook/cache full-row conflict at an on-grid time up to and including origin in this asset-day; conservative provenance flag, not a quote-observation flag.',
        'workbook_cache_filled_mid_equal_origin': 'Exact equality of float32 filled mids at origin under independent workbook/cache reconstructions.',
        'workbook_cache_filled_mid_equal_endpoint': 'Exact equality of float32 filled mids at endpoint under independent workbook/cache reconstructions.',
        'workbook_cache_label_match': 'Exact class agreement under independent workbook/cache float32 reconstructions.',
        'missing_cache_quote_any_endpoint': 'Any missing (NaN/absent row) original bid1/ask1 at either endpoint; does not equate zero with missing.',
        'historical_missing_cache_endpoint': 'Earlier audit definition: any absent/nonfinite original bid1/ask1 at either endpoint. Distinct from the positive, uncrossed P1 rule.',
        'historical_missing_endpoint_kind': 'none, current, future, both according to historical_missing_cache_endpoint; disjoint categories.',
        'equal_filled_mids': 'Exact equality of filled float32 current/future mids; NOT evidence of provider imputation.',
        'historical_missing_equal_filled_flat': 'Historical missing endpoint AND equal filled mids AND legacy_label=1; overlapping subset of historical missing, not an independent exclusion.',
        'candidate_clock': 'origin>=09:10:00 AND endpoint<=14:40:00, with the verified same-day ten-minute target; origins 09:10–14:30 inclusive.',
        'P0_legacy': '1 for every legacy row.',
        'P1_valid_cache_endpoints': 'Both original endpoints have finite positive bid1/ask1 and bid1<=ask1, with same-day ten-minute target. Positive locks retained. Ex-post evaluation eligibility.',
        'P2_candidate_clock': 'P0 restricted only to candidate_clock; independent of quote validity.',
        'P3_intersection': 'P1_valid_cache_endpoints AND P2_candidate_clock. Ex-post evaluation eligibility.',
        'peer_origin_available_cache': 'Only origin_valid_original_cache_quote. Never uses a future endpoint or P1/P3; cache-based availability proxy, not provider release certification.',
        'P1_exclusion_reasons': 'none or pipe-separated endpoint-qualified row_absent, ask1/bid1_missing, nonfinite, nonpositive, crossed; multiple reasons may overlap.',
        'P2_exclusion_reason': 'none, origin_before_0910, or endpoint_after_1440.',
    }
    if name in exact:
        return exact[name]
    prefix = 'origin' if name.startswith('origin_') else 'endpoint'
    rest = name[len(prefix) + 1:]
    subject = 'At ' + prefix + ': '
    if rest.endswith('cache_row_exists'):
        return subject + 'original cache contains this exact date/time row, before reindexing/filling.'
    if rest.endswith('workbook_row_exists'):
        return subject + 'supplied workbook contains this exact normalized date/time row.'
    if rest == 'workbook_cache_status':
        return subject + 'full-row status matched_exact, matched_tolerance, conflict, cache_only, workbook_only, or absent_in_both; numeric tolerance atol=rtol=1e-12.'
    if rest == 'file_value_conflict':
        return subject + '1 for conflict/cache_only/workbook_only, otherwise 0.'
    if rest == 'crossed':
        return subject + 'both original cache quotes finite and bid1>ask1; nonpositive values are separately flagged.'
    if rest == 'locked_positive':
        return subject + 'both original cache quotes finite and positive and bid1==ask1.'
    if rest == 'valid_original_cache_quote':
        return subject + 'original bid1/ask1 finite, strictly positive, and not crossed; locked positive quotes allowed.'
    side = 'ask1' if 'ask1' in rest else 'bid1'
    if rest.endswith('_fill_method'):
        return subject + side + ' original (present), forward_fill (earlier same-day on-grid value), or zero_fill (no earlier present grid value). Zero-valued original cells stay original.'
    if rest.endswith('_source_time'):
        return subject + side + ' supplying same-day grid time HH:MM:SS, or empty for zero_fill; origin fill source never exceeds origin.'
    if rest.startswith('workbook_') and rest.endswith('_present'):
        return subject + side + ' workbook cell nonmissing; not an exchange observation flag.'
    if rest.endswith('_finite_positive'):
        return subject + side + ' original cache cell is finite and >0 before filling.'
    if rest.endswith('_nonpositive'):
        return subject + side + ' original cache cell is finite and <=0 before filling.'
    if rest.endswith('_finite'):
        return subject + side + ' original cache cell is finite before filling.'
    if rest.endswith('_present'):
        return subject + side + ' original cache cell is nonmissing before filling (infinity would be present but nonfinite).'
    raise ValueError('Missing column definition: ' + name)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repository', type=Path, required=True)
    args = ap.parse_args(); root = args.repository; docs = root / DOCS
    checks = json.loads((docs / 'data_checks.json').read_text())
    validation = json.loads((docs / 'validation_checks.json').read_text())
    index = json.loads((docs / 'prediction_index.json').read_text())
    comparison = pd.read_csv(docs / 'workbook_cache_comparison.csv')
    clock = pd.read_csv(docs / 'daily_clock_summary.csv')
    metadata = json.loads((docs / 'workbook_metadata.json').read_text())
    rows = pd.read_csv(root / ARTIFACTS / 'forecast_rows.csv.gz', dtype={'asset_id': str, 'date': str})
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    code_manifest = list(csv.DictReader((root / 'recovery/code_manifest.csv').open()))
    code_verified = all(sha(root / x['export_path']) == x['export_sha256'] and
                        (root / x['export_path']).stat().st_size == int(x['bytes']) for x in code_manifest)
    assert code_verified
    code_check = {'source_sha': SOURCE_SHA, 'export_sha': EXPORT_SHA, 'files_verified': len(code_manifest),
                  'all_source_mapped_code_bytes_unchanged': code_verified,
                  'code_manifest_sha256': sha(root / 'recovery/code_manifest.csv'),
                  'working_scientific_code_modified': False}
    write_json(docs / 'code_identity_check.json', code_check)
    quote_summary = []
    for split, group in rows.groupby('split', sort=True):
        nonpositive = group.filter(regex='_nonpositive$').eq(1).any(axis=1)
        crossed = group.origin_crossed.eq(1) | group.endpoint_crossed.eq(1)
        missing = group.historical_missing_cache_endpoint.eq(1)
        locked = group.origin_locked_positive.eq(1) | group.endpoint_locked_positive.eq(1)
        quote_summary.append({'split': split, 'rows': len(group), 'historical_missing_endpoint': int(missing.sum()),
                              'nonpositive_any_endpoint': int(nonpositive.sum()), 'crossed_any_endpoint': int(crossed.sum()),
                              'extra_P1_exclusions_without_missing': int(((nonpositive | crossed) & ~missing).sum()),
                              'locked_positive_any_endpoint': int(locked.sum()),
                              'locked_positive_retained_P1': int((locked & group.P1_valid_cache_endpoints.eq(1)).sum())})
    pd.DataFrame(quote_summary).to_csv(docs / 'quote_quality_by_split.csv', index=False)
    regimes = []
    for regime, group in clock.groupby('scheduled_session_regime', sort=True):
        regimes.append({'scheduled_session_regime': regime, 'source_cache_dates': int(group.date.nunique()),
                        'asset_days': len(group), 'source_cache_rows': int(group.cache_rows.sum()),
                        'off_legacy_grid_rows': int(group.off_legacy_grid_rows.sum()),
                        'min_rows_per_asset_day': int(group.cache_rows.min()), 'max_rows_per_asset_day': int(group.cache_rows.max())})
    pd.DataFrame(regimes).to_csv(docs / 'clock_regime_summary.csv', index=False)
    calendar = {
        'method': 'source_inspection of existing local reference PDFs; no open-ended calendar search',
        'provider_documentation': 'No provider aggregation/filling/observation/release specification established by inspected workbooks or project notes.',
        'documents': [
            {'relative_reference': 'references/market_data_and_krx/introduction_to_trading_at_krx_stock_market_kofia_2011.pdf',
             'sha256': '04871dede5a51062cd1654510ccad1590f911811226841551b9488a0de3249e4',
             'pdf_pages_1_based': [26, 35], 'status': 'verified',
             'documented_rules': '2011 guide: regular 09:00–15:00; closing call 14:50–15:00. Describes first trading day of year 10:00–15:00 and college-exam day 10:00–16:00 exceptions.',
             'limitation': 'Older rule reference, not a dated exception-session manifest for 2015–2017.'},
            {'relative_reference': 'references/market_data_and_krx/krx_eng_brochure_2017.pdf',
             'sha256': '5b246b91b7135bc76ccd009ac3f293388fe9efe3f62dbe33ce92fb676cad5cf4',
             'pdf_pages_1_based': [16, 30], 'status': 'verified',
             'documented_rules': '2017 KRX brochure records regular-session extension by 30 minutes on 2016-08-01; regular hours shown as 09:00–15:30.'}],
        'row_assignment': 'Schedule-era split at 2016-08-01; exceptional_session_status always unresolved.',
        'exception_dates_and_halts': 'unknown; no flags inferred from gaps, prices, or modern calendars',
        'candidate_clock_scope': 'Predeclared conservative clock only; not executable availability or certified continuous-session membership.',
        'pdf_payloads_published': False,
    }
    write_json(docs / 'calendar_evidence.json', calendar)
    populations = {
        'P0_legacy': 'All 27 assets x 493 common dates x 38 origins.',
        'P1_valid_cache_endpoints': 'Both endpoints originally present, finite positive bid1/ask1, no crossed quotes; same-date exact ten-minute target. Positive locks retained.',
        'P2_candidate_clock': 'Only origin>=09:10 and endpoint<=14:40 and same-date exact +10min: 33 origins 09:10–14:30.',
        'P3_intersection': 'P1 AND P2.',
    }
    splits = [dict(split=name, days=days, first_date=first, last_date=last, rows=count) for name, days, first, last, count in [
        ('train', 345, '2015-07-01', '2016-11-23', 353970),
        ('validation', 49, '2016-11-24', '2017-02-03', 50274),
        ('development_holdout', 99, '2017-02-06', '2017-06-30', 101574)]]
    schema = json.loads((docs / 'schema_columns.json').read_text())
    for name, definition in schema.items():
        definition['csv_type'] = 'integer' if definition['storage_dtype'].startswith('int') else 'string'
        definition['description'] = describe_column(name)
        if definition['csv_type'] == 'integer':
            definition['allowed_values'] = [0, 1, 2] if name == 'legacy_label' else [0, 1]
    write_json(docs / 'schema_columns.json', schema)
    contract = {
        'contract_id': 'qf_august_legacy_review_20261001_v1', 'created_at_utc': now,
        'status': 'reviewable diagnostic contract; legacy loader unchanged',
        'study_version': 'legacy_implemented August; no qf_v2 implementation',
        'source_sha': SOURCE_SHA, 'export_sha': EXPORT_SHA, 'input_commit': INPUT_SHA,
        'input_manifest': 'recovery/input_manifest.csv', 'input_manifest_sha256': sha(root / 'recovery/input_manifest.csv'),
        'input_files': 54, 'input_bytes': 366098160, 'asset_count': 27, 'asset_mapping': 'sorted exact paired filename stems; identifiers treated as strings, no company inference',
        'comparison': {'method': 'source_inspection + recomputed', 'scope': 'exhaustive all 27 pairs, all 688352 rows, all 45 columns',
                       'atol': 1e-12, 'rtol': 1e-12, 'exact_numeric_differences': int(comparison.numeric_exact_mismatches.sum()),
                       'outside_tolerance': int(comparison.numeric_outside_tolerance.sum()),
                       'missingness_mismatches': int(comparison.missingness_mismatches.sum()),
                       'float32_value_mismatches': int(comparison.float32_value_mismatches.sum()),
                       'label_mismatches': checks['workbook_cache_label_mismatches'],
                       'excel_clock_serialization': 'datetime.time -> HH:MM:SS, timedelta -> Python duration string; no modulo-24h conversion, no reclassification as grid observations',
                       'identifier_serialization': 'Workbook numeric identifier/date cells use exact integer strings; source CSV identifier text preserved. No zero-padding or issuer-name mapping inferred.'},
        'three_provenance_layers': {'supplied_file_values': 'presence/validity verified against paired workbook and cache values',
                                    'our_transformations': 'explicit reindexing, per-asset/day forward fill then zero fill; quote source time and method saved',
                                    'provider_process': 'unknown aggregation, provider filling, observation time, release time, units and adjustment policy'},
        'local_clock_timezone_convention': 'Asia/Seoul', 'provider_timestamp_semantics': 'unknown', 'provider_operational_availability': 'unknown',
        'units': 'Original file numeric scale preserved; no provider unit certification or corporate-action adjustment established.',
        'keys': ['asset_id', 'date', 'origin_time'], 'endpoint': 'endpoint_time on same date',
        'ordering': 'lexicographic asset_id,date,origin_time', 'date_format': 'YYYY-MM-DD', 'time_format': 'HH:MM:SS',
        'common_dates': 493, 'excluded_union_dates': ['2015-07-28'],
        'grid': GRID, 'origins_per_asset_day': 38,
        'alignment': 'exact clock string after whitespace stripping; off-grid rows never seed fills; duplicate on-grid keys rejected by diagnostic',
        'filling': 'per asset and date, forward fill each column using earlier on-grid values; remaining gaps zero fill; no backward fill, interpolation, or cross-day history',
        'labels': {'formula': 'ask/bid arrays float32 before mid=(ask1+bid1)/2.0; float32 r=(mid_next-mid)/mid when mid>0',
                   'threshold': 0.0001, 'threshold_comparison': 'original NumPy float32 scalar versus Python float epsilon; strict inequalities',
                   'class_order': {'0': 'Down if r < -0.0001', '1': 'Flat otherwise, including equality and nonpositive current mid', '2': 'Up if r > +0.0001'},
                   'horizon': 'next fixed-grid row, verified same-day exactly ten minutes',
                   'rounding': 'No explicit rounding; original float32 arithmetic retained. Not sign-only labels.',
                   'source': 'lob_forecasting/cli/parallel.py:bake_tensors:78-129'},
        'features': {'main_feature_set': 'no_investor', 'input_width': 13,
                     'implementation': 'log mid; log ask1/mid and bid1/mid; log1p ask/bid quantities levels 1–5, epsilon1e-9, final torch.nan_to_num',
                     'source': ['lob_forecasting/experiments/qf_training.py:84', 'lob_forecasting/data/transforms.py:10-37', 'lob_forecasting/cli/parallel.py:124-125'],
                     'audit_scope': 'Top-quote labels/provenance reconstructed without importing torch or building training tensors.'},
        'splits': splits, 'development_holdout_is_untouched_lockbox': False,
        'populations': populations, 'population_totals': checks['population_totals'],
        'evaluation_selection_warning': 'P1/P3 use future endpoint quality and are ex-post evaluation populations, not executable origin-time selection rules.',
        'peer_availability_rule': 'Origin only; peer_origin_available_cache = origin_valid_original_cache_quote. Do not pass future-endpoint or P1/P3 eligibility into peer inputs.',
        'upstream_observation_warning': 'Nonmissing workbook cell does not prove observed exchange quote; equal values do not prove imputation.',
        'schema_path': DOCS + '/schema_columns.json', 'schema': schema,
        'row_key_sha256': checks['row_key_sha256'], 'row_key_label_sha256': checks['row_key_label_sha256'],
        'digest_serialization': checks['digest_serialization'], 'calendar_evidence': DOCS + '/calendar_evidence.json',
        'prediction_index': DOCS + '/prediction_index.json', 'prediction_alignment_checks': DOCS + '/validation_checks.json',
    }
    write_json(docs / 'data_contract.json', contract)
    schema_text = '# Forecast-row schema\n\nUTF-8 CSV inside deterministic gzip; empty source_time means zero_fill. '
    schema_text += 'Read identifiers and all date/time fields as strings, with `keep_default_na=False` if preserving empty strings. '
    schema_text += 'Flags are integers 0/1. Provider observation/release semantics remain unknown.\n\n'
    schema_text += table([{'column': k, 'type': v['csv_type'], 'meaning': v['description']} for k, v in schema.items()], ['column', 'type', 'meaning']) + '\n'
    (docs / 'schema.md').write_text(schema_text)
    def population_table(name, grouping):
        frame = pd.read_csv(docs / name); records = []
        for record in frame.to_dict('records'):
            row = {c: record[c] for c in grouping}
            row.update(population=record['population'], retained=record['retained'], excluded=record['excluded'],
                       **{c + '_pct': '{:.4f}'.format(100 * record[c + '_proportion']) for c in ['down', 'flat', 'up']})
            records.append(row)
        return table(records, grouping + ['population', 'retained', 'excluded', 'down_pct', 'flat_pct', 'up_pct'])
    selected_table = table([{'model': x['model'], 'seed': x['seed'], 'run_id': x['run_id'], 'epoch': x['selected_epoch_recorded'],
                             'recorded_val_NLL': '{:.9f}'.format(x['recorded_selected_validation_log_loss']),
                             'key_label_errors': x['unmatched_keys'] + x['label_mismatches']} for x in validation['prediction_alignment']],
                           ['model', 'seed', 'run_id', 'epoch', 'recorded_val_NLL', 'key_label_errors'])
    report = f'''# August KRX data contract and forecast-row review

Created {now}. Evidence statuses below are **verified**, **inferred**, or **unknown**; verification methods are source_inspection, static_code, saved_artifact, and recomputed. This is the accepted August baseline. No September recovery, qf_v2 work, training, checkpoint loading, inference, cache conversion, or model-score selection was performed.

All 54 existing inputs match the pinned manifest by bytes, ordinary Git blob ID, and SHA-256, before and after read-only work. All 27 workbook/cache pairs were fully compared: 688,352 rows and 45 columns per pair schema (4 identifiers, 41 numeric columns). The 505,818-row diagnostic and nine frozen selected prediction files are ready for independent review through Git. Source-mapped code is unchanged ({len(code_manifest)} files verified).

## Provenance and boundaries

- Verified remote: `{REMOTE}`; task branch `{BRANCH}` is rooted in the published code-only export `{EXPORT_SHA}`. Original committed source is `{SOURCE_SHA}`. Input bytes are already published at `{INPUT_SHA}`; no need to publish those 366,098,160 bytes again.
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

{table(splits, ['split', 'days', 'first_date', 'last_date', 'rows'])}

## Named masks, retained counts and class proportions

P0 is every legacy row. P1 requires original finite strictly positive bid1/ask1 at both endpoints, no cross, and same-day exact +10min; positive locks are retained. P2 applies only the predeclared origin>=09:10 / endpoint<=14:40 clock (33 origins through 14:30). P3 is their intersection. **P1/P3 are ex-post evaluation populations.** A future label endpoint must never control current peer inputs; the published `peer_origin_available_cache` uses only that peer's origin quote. It remains a cache-quality proxy, not a provider-certified availability flag. No legacy rows are deleted.

Totals: P0 **505,818**; P1 **469,391** (36,427 excluded); P2 **439,263** (66,555 excluded); P3 **432,560** (73,258 excluded). Class percentages are within each retained population. Excluded counts always use the corresponding P0 group as denominator.

{population_table('population_counts_by_split.csv', ['split'])}

All 27 assets and populations are shown below; exact class counts and proportions are in [by asset](population_counts_by_asset.csv). Joint asset × split × population counts are in [asset/split table](population_counts_by_asset_split.csv), with [schedule-era/split](population_counts_by_regime_split.csv) and [origin/split](population_counts_by_origin_split.csv) support.

{population_table('population_counts_by_asset.csv', ['asset_id'])}

## Missingness and price quality

Recomputed historical missing-cache-endpoint result is **34,600 / 505,818 = {34600/505818*100:.6f}%** (train 28,195; validation 1,994; development holdout 4,411). Current-only=9,365; future-only=20,719; both=4,516. Historical missing means absent/nonfinite original top quotes; there is no infinity-only discrepancy here. Missing endpoint AND equal filled mids AND Flat is **27,366 / 505,818 = {27366/505818*100:.6f}%**, an overlapping subset, not a second disjoint category. Equal prices alone never identify imputation.

P1 excludes **1,827 additional rows** without historical missingness because of nonpositive or crossed values. The following row counts may overlap and must not be summed as independent causes. Positive locks can coexist with a different endpoint defect; only that defect rejects the row.

{table(quote_summary, ['split', 'rows', 'historical_missing_endpoint', 'nonpositive_any_endpoint', 'crossed_any_endpoint', 'extra_P1_exclusions_without_missing', 'locked_positive_any_endpoint', 'locked_positive_retained_P1'])}

Per-top-quote presence/finite/positive flags, source time, forward-fill/zero-fill/original method, crossing/locking, and full file-value provenance are in every forecast row. [Missingness by asset and split](missingness_by_asset_split.csv) includes per-endpoint and per-quote counts. Origin quote filling: each side has 11,254 forward-filled and 2,627 zero-filled occurrences. Future endpoint: each side has 22,717 forward-filled and 2,518 zero-filled occurrences. A literal zero supplied in the file remains `original`, not zero_fill. These are forecast endpoint occurrences, not unique original-row counts.

## Clock, session regimes and unknown provider semantics

Source/cache rows total 688,352, of which 170,601 are off the imposed legacy grid. Daily counts range 18–55. The 493 common-date population is missing 1,844 asset/grid rows (1,846 if counting all source dates); existing grid rows can separately contain missing quote cells. The input anomaly clocks remain excluded only by the faithful legacy grid. [Daily clock summary](daily_clock_summary.csv) preserves coverage and off-grid counts; [regime summary](clock_regime_summary.csv) is split at 2016-08-01.

Existing local KOFIA 2011 guide PDF pages 26/35 documents 09:00–15:00 regular hours, closing call 14:50–15:00, and first-year-day/exam-day exceptions. Existing KRX 2017 brochure PDF pages 16/30 documents the 30-minute extension on 2016-08-01 and 09:00–15:30 regular hours. Hashes and scope are in [calendar evidence](calendar_evidence.json). Those documents support schedule eras but do not identify every 2015–2017 exceptional session in these files. No halt/auction/exception flags were invented. The historical fixed grid crosses different official schedules; candidate_clock is a provisional restriction, not a certification of continuous trading or executable timing. Local-clock convention is Asia/Seoul; provider timestamp/release semantics remain unknown.

## Frozen historical predictions

The index was frozen at **{index['frozen_at_utc']}**, before population creation and before any restricted scoring; no restricted model scores were computed in this task. Selection follows the existing explicit mapping in `lob_forecasting/experiments/discrimination_finalize.py:16` and the user's pinned paths. All nine originals have 50,274 validation and 101,574 development-holdout rows (151,848 each), despite the filename `test_predictions.csv.gz`. Probabilities map Down/Flat/Up to columns `probability_down/flat/up`, are finite in [0,1], and have maximum row-sum error <=1.4e-7. Stable keys, all labels and both splits agree exactly with this contract; duplicate/missing keys are zero. Prediction endpoints are joined from the verified +10min contract because original predictions store origins only.

{selected_table}

Table NLL values are **existing training-history entries at the recorded checkpoint**, not newly computed restricted scores. Checkpoint selection was independently replayed from every saved history using validation log loss and strict min_delta=0.0001 (`discrimination_runner.py:53-62`); all nine selected epochs match. Original probability bytes are preserved; compressed and decompressed hashes, configs, histories, manifests and status are in [frozen index](prediction_index.json) and [checks](validation_checks.json). No checkpoint payload is included. Existing flags in predictions that say `unavailable` are preserved; use the diagnostic contract for cache-derived flags, not an invented observation flag.

**UC metadata conflict:** `F_best_UC_42_stage2` epoch151 equals `F_PAIR_5_SHALLOW_INTERACTION_UC_42` in decompressed predictions/config/history/manifest and is counted once. The legacy selection YAML and Stage-2 decision instead name PAIR_0_BASELINE. We preserve both records and explicitly use the pinned finalizer path; this does not resolve the historical prose inconsistency or justify a new winner. Some PAIR names share identical numeric config, so config equality alone is not artifact identity. Per-run provenance remains separate from configured architecture. The old pilot family is not substituted.

UM_V2_DEPTH is verified in all selected manifests (epochs128/122/127). The corrected implementation uses leave-target-out contemporaneous/backward factors, log1p total five-level depth, train-only .001/.999 clipping and training mean/std (`discrimination_factors.py:7-45`). It is not the old unscaled UM. Seeds7/123 are matching selected Stage-2 artifacts, not new runs or an ensemble. This report makes no September-anchor or untouched-test claim.

## Checks, readiness and receiver action

Executed: exact input hashes before/after; the inspected recovery helper's read-only verify-only check of all 54 paths against the published commit tree/blob identities; exhaustive workbook/cache keys/schema/values/missingness; all asset-day literal float32 label equivalence; unique keys and split boundaries; synthetic strict-threshold/nonpositive/fill/off-grid/cross/lock/clock/future-peer checks; nine saved probability/key/label/split checks; nine history-based epoch replays; original code-manifest hash checks; syntax checks of task helpers. The initial optional `pdftotext` attempt found the executable absent; existing `pypdf` successfully read the local references. Missing-path search probes were corrected by reading the actual repository paths. An optional report-wording probe failed because it required a literal word rather than testing data; it is not part of scientific validation. No environment changes were needed.

Data-ready means the diagnostic contract and masks are verified, not that provider semantics or a new scientific experiment are approved. Predictions-ready means these nine frozen files support independent rescoring, not that this session reproduced model training. No packages are missing for this CPU task. Pending: provider semantics; exact dated exception/halt calendar if needed later; UC selection-description conflict; historical per-run execution identity. Legacy Stage-2 hardcoded paths remain unchanged and would require separate runtime work before future training.

The receiving RTX3090 session should verify the advertised task SHA, fetch this exact branch into a separate worktree, read [handoff.json](handoff.json), validate its file hashes, then join probabilities to masks by the explicit key mapping in the contract. Report P0 first and P1/P2/P3 separately, with validation and development_holdout separate, per seed; preserve the frozen selections and do not construct a probability ensemble by default. Any rescoring must document natural log, normalization/clipping and row weighting. The legacy trainer normalized probabilities and called sklearn log_loss (`qf_training.py:30-31`, `evaluation/qf_metrics.py:10-12`); the old finalizer used its own direct loss helper. Library-dependent clipping defaults are not assumed identical. No scores were recomputed here.

Inputs are reusable from the published input SHA and `recovery/input_manifest.csv`. Git transfers committed mask/prediction artifacts in this task; it does not automatically transfer any other untracked files. This handoff is a candidate comparison baseline and does not authorize overwriting receiving-server work. No unavailable-server action is required.

Run instructions and precise file formats are in [schema.md](schema.md), [data_contract.json](data_contract.json), and [receiver.md](receiver.md). The final published SHA is delivered outside these committed files, avoiding self-reference.
'''
    (docs / 'report.md').write_text(report)
    receiver = f'''# RTX3090 receiver instructions

Read `{DOCS}/handoff.json` at `refs/heads/{BRANCH}` after verifying the full advertised SHA supplied in the publication receipt. Preserve your own newer/dirty work; use a new isolated worktree. The task's parent is the published August code-only root `{EXPORT_SHA}`. No old scientific history is needed for masks/predictions.

```sh
export GIT_LFS_SKIP_SMUDGE=1
git -C YOUR_VERIFIED_BRIDGE fetch --filter=blob:none --no-tags --no-recurse-submodules origin refs/heads/{BRANCH}
git -C YOUR_VERIFIED_BRIDGE rev-parse FETCH_HEAD
# Check against DATA_HANDOFF_SHA before the next command.
git -C YOUR_VERIFIED_BRIDGE worktree add --detach NEW_DATA_REVIEW_DIRECTORY FETCH_HEAD
```

All new task payloads total about 32 MB; no checkpoint payload is present. Historical code-only ancestry is small. No blanket historical checkout or unfiltered clone is needed. If this bridge's remote is not the verified `{REMOTE}`, stop and resolve identity before fetching. Do not change an existing remote blindly.

First verify the committed packet without any writes or third-party packages:

```sh
python3 -B scripts/qf_data_review/verify_handoff.py --repository .
```

The receiver may then run the inspected bounded CPU validator with an existing Python >=3.9 containing compatible numpy/pandas/openpyxl (recorded versions are in handoff.json; NumPy scalar-promotion behavior is part of the legacy arithmetic and must not be silently changed):

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -B scripts/qf_data_review/validate_review.py --repository .
```

This writes only the task validation JSON, deterministically from supplied artifacts, and performs no model scoring, inference, cache conversion or training. Avoid overwriting independently modified task results: run from a fresh task worktree. Source workbooks are not required for rescoring already-supplied masks/predictions.

To independently reproduce the full audit, use a fresh isolated copy of the exported code and copy/import the frozen replay inputs/index first. `build_data_review.py --inputs EXISTING_VERIFIED_INPUT_ROOT --repository NEW_AUDIT_ROOT` compares every workbook and creates new diagnostic masks. It refuses to overwrite forecast_rows.csv.gz. `freeze_predictions.py` is only for a destination without replay_inputs; it requires an existing Stage-2 artifact root, and is unnecessary on the receiver because all needed artifacts are already committed here. Then run validate_review.py and finalize_packet.py. Never execute a legacy queue/training launcher for this verification.

Existing inputs: inspect `recovery/materialize_inputs.py` before use. Its `--verify-only` mode checks a matching local input root; it writes nothing. If missing, use the documented blob-filtered fetch of `{INPUT_SHA}` and restore only its 54 manifest-listed files to a NEW destination. Those files are already-published ordinary Git blobs. No cache rebuilding or checkpoint downloading is needed.

Join keys: prediction asset -> asset_id (preserve _t); prediction date YYYYMMDD -> date YYYY-MM-DD; timestamp -> origin_time. Check a one-to-one outer join before filtering; endpoint_time is same-date +10min from the contract. Each run has 151,848 legacy evaluation rows. Select the original split first, then the named population; report each population's support separately. Predictions are saved probabilities, not logits. Class order is [Down,Flat,Up]=[0,1,2]. No seed ensembling or new checkpoint selection is authorized by this handoff.

P1/P3 eligibility uses the future endpoint and must never become an origin-time peer-input mask. Use only peer_origin_available_cache if inspecting origin cache quality, while retaining unknown provider release timing. Provider aggregation/filling semantics and the precise exceptional-day calendar remain unresolved.
'''
    (docs / 'receiver.md').write_text(receiver)
    files = []
    for directory in ['scripts/qf_data_review', DOCS, ARTIFACTS]:
        for path in sorted((root / directory).rglob('*')):
            if path.is_file() and path.name != 'handoff.json':
                entry = {'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': sha(path)}
                if path.name == 'forecast_rows.csv.gz':
                    digest = hashlib.sha256()
                    with gzip.open(path, 'rb') as stream:
                        for block in iter(lambda: stream.read(1 << 20), b''):
                            digest.update(block)
                    entry['decompressed_sha256'] = digest.hexdigest()
                files.append(entry)
    ready = bool(checks['data_ready'] and validation['real_key_split_label_mask_checks_passed'] and code_verified)
    handoff = {'created_at_utc': now, 'remote': REMOTE, 'branch': BRANCH,
               'source_sha': SOURCE_SHA, 'export_sha': EXPORT_SHA, 'input_sha': INPUT_SHA,
               'report_commit_sha': 'Supplied in external publication receipt; intentionally not self-referenced here.',
               'prior_report_shas': ['a323db8aea04fbb33b8e32e7644647106150899c', '2912890319b19a731e6e446b486ff7c8c6f3c3e3'],
               'task_scope': 'August data identity, exhaustive workbook/cache comparison, diagnostic forecast masks and frozen selected historical prediction transfer',
               'host': {'session_nickname': 'A5000', 'actual_devices_recorded_once': '4 NVIDIA GeForce RTX 4090',
                        'vram_mib_each': 24564, 'driver': '595.71.05', 'execution_device': 'CPU', 'platform': platform.system()},
               'dependencies': {k: checks[k] for k in ['python', 'numpy', 'pandas', 'openpyxl']},
               'environment_modified': False, 'training_or_inference_run': False,
               'source_mapped_code_unchanged': code_verified, 'historical_dirty_state_caveat': 'Original backup had unrelated mode/reference deltas; this task uses committed code-only export; per-run SHA/dirty state unknown.',
               'input_manifest': 'recovery/input_manifest.csv', 'input_manifest_sha256': sha(root / 'recovery/input_manifest.csv'),
               'code_manifest': 'recovery/code_manifest.csv', 'code_manifest_sha256': sha(root / 'recovery/code_manifest.csv'),
               'inputs': {'files': 54, 'workbooks': 27, 'csv_caches': 27, 'bytes': 366098160, 'all_identity_checks_pass': True,
                          'reused_matching_local_files': True, 'new_input_downloads': 0, 'new_workbook_cache_publications': 0,
                          'materializable_paths': ['data/raw/KRXaggData/KR*_t.xlsx', 'data/processed/KR*_t.csv'],
                          'exact_path_blob_checksums': 'recovery/input_manifest.csv', 'commit': INPUT_SHA},
               'forecast_rows': ARTIFACTS + '/forecast_rows.csv.gz', 'row_count': checks['forecast_rows'],
               'row_key_sha256': checks['row_key_sha256'], 'row_key_label_sha256': checks['row_key_label_sha256'],
               'split_digests': checks['split_digests'], 'digest_serialization': checks['digest_serialization'],
               'schema': {'path': DOCS + '/schema_columns.json', 'contract': DOCS + '/data_contract.json',
                          'row_key': ['asset_id', 'date', 'origin_time'], 'endpoint': 'endpoint_time', 'class_order': ['Down', 'Flat', 'Up']},
               'populations': populations, 'population_totals': checks['population_totals'],
               'selected_prediction_index_path': DOCS + '/prediction_index.json',
               'selected_prediction_index_sha256': sha(docs / 'prediction_index.json'),
               'selected_prediction_index': [{k: x[k] for k in ['model', 'seed', 'run_id', 'selected_epoch', 'prediction_path', 'prediction_sha256', 'decompressed_sha256']} for x in index['selected']],
               'selection_conflict': index['selection_caveat'],
               'readiness': {'data_ready': ready, 'predictions_ready': 'true' if validation['predictions_ready'] else 'partial',
                             'provider_semantics_ready': False, 'exception_calendar_ready': False,
                             'new_training_pipeline_validated': False, 'independent_rescoring_inputs_ready': ready and validation['predictions_ready']},
               'missing_prediction_files': index['missing'], 'missing_cpu_dependencies': [],
               'unknowns': ['Provider aggregation/filling/observation/release timing and units', 'Dated exceptional sessions and halts',
                            'Historical per-run SHA/dirty state/host', 'UC YAML/decision versus explicit selected artifact inconsistency'],
               'report_path': DOCS + '/report.md', 'receiver_path': DOCS + '/receiver.md',
               'task_file_hashes_exclude': [DOCS + '/handoff.json'], 'task_files': files,
               'payload_bytes_excluding_handoff': sum(x['bytes'] for x in files),
               'next_action': 'RTX3090 fetch exact advertised ref/SHA into isolated worktree; verify task hashes and row joins, then independently rescore frozen files per split and named population without selecting new winners.'}
    write_json(docs / 'handoff.json', handoff)
    print(json.dumps({'data_ready': ready, 'predictions_ready': handoff['readiness']['predictions_ready'],
                      'files_hashed': len(files), 'payload_bytes': handoff['payload_bytes_excluding_handoff']}))


if __name__ == '__main__':
    main()
