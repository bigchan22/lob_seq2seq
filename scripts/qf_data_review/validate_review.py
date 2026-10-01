"""Bounded CPU checks of the row contract and frozen predictions, without scoring."""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_data_review import ASK, BID, GRID, KEYS, align_quotes, compare_pair, key_digest, literal_labels, reconstruct, sha


def synthetic_checks():
    dates = ['20150101', '20150102']
    records = [
        ('20150101', '08:50:00', 900, 900),
        ('20150101', '09:10:00', 100, 100),
        ('20150101', '09:20:00', 99, 101),
        ('20150101', '09:30:00', np.nan, 100),
        ('20150102', '08:50:00', 900, 900),
        ('20150102', '09:20:00', 200, 199),
    ]
    raw = pd.DataFrame(records, columns=['ORD_DD', 'TIME_INTERVAL', ASK, BID])
    raw['ISU_CD'] = 'SYNTHETIC'; raw['TICKER'] = '000001'
    compared, _, _, _ = compare_pair('SYNTHETIC', raw, raw, 1e-12, 1e-12)
    aligned, mid, labels, source = align_quotes(compared, dates)
    assert mid.dtype == np.float32
    assert mid[0, 0] == 0 and mid[1, 0] == 0 and mid[1, 1] == 0
    assert mid[0, 1] == 100 and mid[0, 3] == 99.5
    assert source['ask1']['method'][3] == 'forward_fill'
    assert source['ask1']['source_time'][3] == '09:20:00'
    assert source['ask1']['method'][39] == 'zero_fill'
    rows = reconstruct('SYNTHETIC', compared, raw, dates, {day: 'train' for day in dates})
    r = rows[(rows.date == '2015-01-01') & (rows.origin_time == '09:10:00')].iloc[0]
    assert r.origin_locked_positive == 1 and r.origin_valid_original_cache_quote == 1
    assert r.endpoint_crossed == 1 and r.P1_valid_cache_endpoints == 0
    assert r.peer_origin_available_cache == 1
    assert rows.candidate_clock.sum() == 66
    assert set(rows.loc[rows.candidate_clock == 1, 'origin_time']) == set(GRID[1:34])
    changed = raw.copy()
    changed.loc[changed.TIME_INTERVAL == '09:20:00', [ASK, BID]] = np.nan
    changed_compared, _, _, _ = compare_pair('SYNTHETIC', changed, changed, 1e-12, 1e-12)
    future_changed = reconstruct('SYNTHETIC', changed_compared, changed, dates, {day: 'train' for day in dates})
    rr = future_changed[(future_changed.date == '2015-01-01') & (future_changed.origin_time == '09:10:00')].iloc[0]
    assert rr.peer_origin_available_cache == r.peer_origin_available_cache
    assert rr.endpoint_cache_ask1_present == 0
    assert literal_labels(np.array([10000, 10001], dtype=np.float32)).tolist() == [1]
    assert literal_labels(np.array([10000, 9999], dtype=np.float32)).tolist() == [1]
    assert literal_labels(np.array([10000, 10002], dtype=np.float32)).tolist() == [2]
    assert literal_labels(np.array([10000, 9998], dtype=np.float32)).tolist() == [0]
    assert literal_labels(np.array([0, 100], dtype=np.float32)).tolist() == [1]
    assert literal_labels(np.array([-100, 100], dtype=np.float32)).tolist() == [1]
    # Exact mathematical +/-1bp computed in float64 equals the Python epsilon.
    assert literal_labels(np.array([10000, 10001], dtype=np.float64)).tolist() == [1]
    assert literal_labels(np.array([10000, 9999], dtype=np.float64)).tolist() == [1]
    duplicate = pd.concat([compared, compared.iloc[[1]]], ignore_index=True)
    try:
        align_quotes(duplicate, dates)
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate on-grid keys were not rejected')
    assert rows.no_cross_day_target.eq(1).all() and rows.exact_ten_minute_target.eq(1).all()
    return ['off-grid rows do not seed fills', 'leading gaps zero-filled', 'same-day per-quote forward fill',
            'no cross-day fill or target', 'float32 +/-1bp and strict equality boundaries',
            'nonpositive current midpoint defaults Flat', 'positive locks retained', 'crossed quotes rejected',
            'future endpoint cannot alter current peer availability', '33 candidate origins per day',
            'duplicate on-grid keys rejected']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repository', required=True, type=Path)
    args = ap.parse_args(); root = args.repository
    docs = root / 'docs/qf_data_review/20261001'
    tested = synthetic_checks()
    for path in (root / 'scripts/qf_data_review').glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    index_path = docs / 'prediction_index.json'
    index_hash = sha(index_path); index = json.loads(index_path.read_text())
    rows = pd.read_csv(root / 'artifacts/qf_data_review/20261001/forecast_rows.csv.gz',
                       dtype={'asset_id': str, 'date': str, 'origin_time': str, 'endpoint_time': str})
    assert not rows.duplicated(KEYS).any()
    assert rows[KEYS].equals(rows.sort_values(KEYS, kind='stable')[KEYS])
    checks = json.loads((docs / 'data_checks.json').read_text())
    assert key_digest(rows) == checks['row_key_label_sha256']
    for split, days, first, last in [('train', 345, '2015-07-01', '2016-11-23'),
                                   ('validation', 49, '2016-11-24', '2017-02-03'),
                                   ('development_holdout', 99, '2017-02-06', '2017-06-30')]:
        part = rows[rows.split == split]
        assert part.date.nunique() == days and part.date.min() == first and part.date.max() == last
    origin = pd.to_datetime(rows.date + ' ' + rows.origin_time)
    endpoint = pd.to_datetime(rows.date + ' ' + rows.endpoint_time)
    assert (endpoint - origin).eq(pd.Timedelta(minutes=10)).all()
    expected_p1 = rows.origin_valid_original_cache_quote.eq(1) & rows.endpoint_valid_original_cache_quote.eq(1)
    expected_p2 = rows.origin_time.ge('09:10:00') & rows.endpoint_time.le('14:40:00')
    assert rows.P0_legacy.eq(1).all()
    assert rows.P1_valid_cache_endpoints.eq(expected_p1.astype(int)).all()
    assert rows.P2_candidate_clock.eq(expected_p2.astype(int)).all()
    assert rows.P3_intersection.eq((expected_p1 & expected_p2).astype(int)).all()
    assert rows.peer_origin_available_cache.eq(rows.origin_valid_original_cache_quote).all()
    expected = rows[rows.split.isin(['validation', 'development_holdout'])][KEYS + ['endpoint_time', 'split', 'legacy_label']]
    validations = []
    for record in index['selected']:
        path = root / record['prediction_path']
        assert sha(path) == record['prediction_sha256']
        pred = pd.read_csv(path, dtype={'asset': str, 'date': str, 'timestamp': str})
        pred = pred.rename(columns={'asset': 'asset_id', 'timestamp': 'origin_time', 'split': 'prediction_split'})
        pred['date'] = pd.to_datetime(pred.date, format='%Y%m%d').dt.strftime('%Y-%m-%d')
        assert not pred.duplicated(KEYS).any()
        joined = expected.merge(pred, on=KEYS, how='outer', validate='one_to_one', indicator=True)
        unmatched = int(joined['_merge'].ne('both').sum())
        label_errors = int(joined.legacy_label.ne(joined.true_class).sum())
        split_errors = int(joined.split.ne(joined.prediction_split).sum())
        directory = path.parent
        history = pd.read_csv(directory / 'training_history.csv')
        best, epoch = float('inf'), None
        for row in history.itertuples(index=False):
            if row.validation_log_loss < best - record['configuration']['min_delta']:
                best, epoch = row.validation_log_loss, int(row.epoch)
        status = json.loads((directory / 'status.json').read_text())
        validation = {'model': record['model'], 'seed': record['seed'], 'run_id': record['run_id'],
                      'rows': len(pred), 'unmatched_keys': unmatched, 'label_mismatches': label_errors,
                      'split_mismatches': split_errors, 'selected_epoch_recorded': record['selected_epoch'],
                      'selected_epoch_replayed_from_history': epoch,
                      'selection_metric': 'validation_log_loss with configured strict min_delta improvement',
                      'recorded_selected_validation_log_loss': best,
                      'status_record': status.get('state'),
                      'row_key_label_sha256': key_digest(joined.sort_values(KEYS, kind='stable')),
                      'endpoint_key_provenance': 'prediction stores origin; endpoint joined from verified legacy +10min same-day contract',
                      'restricted_scores_computed': False}
        validations.append(validation)
    result = {'synthetic_checks_passed': tested, 'real_key_split_label_mask_checks_passed': True,
              'prediction_alignment': validations, 'prediction_index_unchanged': sha(index_path) == index_hash,
              'predictions_ready': len(validations) == 9 and all(
                  not (v['unmatched_keys'] or v['label_mismatches'] or v['split_mismatches'])
                  and v['selected_epoch_recorded'] == v['selected_epoch_replayed_from_history']
                  and v['status_record'] == 'SUCCEEDED' for v in validations),
              'training_or_inference_run': False, 'restricted_model_scores_computed': False}
    (docs / 'validation_checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
