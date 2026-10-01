"""Freeze already-selected August Stage-2 replay inputs; never score or load models."""
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

RUNS = {
    'UA': {42: 'F_PAIR_2_DOUBLE_LR_UA_42', 7: 'F_best_UA_7', 123: 'F_best_UA_123'},
    'UC': {42: 'F_best_UC_42_stage2', 7: 'F_best_UC_7', 123: 'F_best_UC_123'},
    'UM_V2_DEPTH': {42: 'D_UM_V2_DEPTH_42', 7: 'D_selected_7', 123: 'D_selected_123'},
}
PROBS = ['probability_down', 'probability_flat', 'probability_up']


def sha(path, decompress=False):
    digest = hashlib.sha256()
    with (gzip.open(path, 'rb') if decompress else path.open('rb')) as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage-root', required=True, type=Path)
    ap.add_argument('--repository', required=True, type=Path)
    args = ap.parse_args()
    root = args.repository
    docs = root / 'docs/qf_data_review/20261001'
    output = root / 'artifacts/qf_data_review/20261001/replay_inputs'
    docs.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=False)
    selected, missing, files = [], [], []
    for model, seeds in RUNS.items():
        for seed, run_id in seeds.items():
            source = args.stage_root / 'runs' / run_id
            required = ['test_predictions.csv.gz', 'resolved_config.json', 'manifest.json', 'training_history.csv', 'status.json']
            absent = [name for name in required if not (source / name).is_file()]
            if absent:
                missing.extend({'model': model, 'seed': seed, 'run_id': run_id, 'missing': name} for name in absent)
                continue
            prediction = source / required[0]
            frame = pd.read_csv(prediction, dtype={'date': str, 'timestamp': str, 'asset': str})
            config = json.loads((source / 'resolved_config.json').read_text())
            manifest = json.loads((source / 'manifest.json').read_text())
            assert config['seed'] == seed and manifest['model'] == model and manifest['seed'] == seed
            assert set(frame.seed) == {seed}
            assert set(frame.model_id) == {model}
            assert set(frame.split) == {'validation', 'development_holdout'}
            assert set(frame.feature_set) == {'no_investor'}
            assert set(frame.true_class) <= {0, 1, 2}
            assert not frame.duplicated(['asset', 'date', 'timestamp']).any()
            probabilities = frame[PROBS].to_numpy(dtype=np.float64)
            assert np.isfinite(probabilities).all() and (probabilities >= 0).all() and (probabilities <= 1).all()
            assert np.max(np.abs(probabilities.sum(1) - 1)) < 1e-5
            epochs = sorted(int(x) for x in frame.validation_selected_checkpoint_epoch.unique())
            assert len(epochs) == 1
            if model == 'UA' and seed == 42:
                assert epochs == [178]
            if model == 'UC' and seed == 42:
                assert epochs == [151]
            destination = output / '{}_seed{}'.format(model, seed)
            destination.mkdir()
            for name in required:
                target = destination / name
                shutil.copyfile(source / name, target)
                identity = {'path': str(target.relative_to(root)), 'original_relative_path': 'runs/' + run_id + '/' + name,
                            'bytes': target.stat().st_size, 'sha256': sha(target)}
                assert identity['sha256'] == sha(source / name)
                if name.endswith('.gz'):
                    identity['decompressed_sha256'] = sha(target, True)
                files.append(identity)
            counts = []
            for split, group in frame.groupby('split', sort=True):
                counts.append({'split': split, 'rows': len(group), 'days': int(group.date.nunique()),
                               'assets': int(group.asset.nunique()), 'first_date': group.date.min(), 'last_date': group.date.max()})
            record = {'model': model, 'seed': seed, 'family': 'selected_august_stage2', 'run_id': run_id,
                      'original_relative_path': 'runs/' + run_id,
                      'selected_epoch': epochs[0], 'configuration_sha256': sha(source / 'resolved_config.json'),
                      'configuration_canonical_sha256': hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                      'configuration': config, 'split_counts': counts, 'probability_columns': PROBS,
                      'class_mapping': {'0': 'Down', '1': 'Flat', '2': 'Up'}, 'protocol_ids': sorted(frame.protocol_id.unique().tolist()),
                      'prediction_path': str((destination / required[0]).relative_to(root)),
                      'prediction_sha256': sha(prediction), 'decompressed_sha256': sha(prediction, True),
                      'max_probability_sum_error': float(np.max(np.abs(probabilities.sum(1) - 1))),
                      'historical_code_sha': 'unknown per-run; linked by recovered Stage-2 finalizer/source family',
                      'historical_dirty_state': 'unknown', 'historical_execution_host': 'unknown'}
            if model == 'UC' and seed == 42:
                duplicate = args.stage_root / 'runs/F_PAIR_5_SHALLOW_INTERACTION_UC_42'
                assert sha(duplicate / 'test_predictions.csv.gz', True) == record['decompressed_sha256']
                assert all(sha(duplicate / name) == sha(source / name) for name in ['resolved_config.json', 'training_history.csv', 'manifest.json'])
                record['duplicate_of'] = 'runs/F_PAIR_5_SHALLOW_INTERACTION_UC_42'
                record['duplicate_counted_once'] = True
            selected.append(record)
    provenance = output / 'selection_provenance'
    provenance.mkdir()
    for name in ['selected_configurations.yaml', 'selected_configuration_hashes.txt', 'STAGE2_DECISION.json']:
        source = args.stage_root / name
        if source.is_file():
            target = provenance / name
            shutil.copyfile(source, target)
            files.append({'path': str(target.relative_to(root)), 'original_relative_path': name,
                          'bytes': target.stat().st_size, 'sha256': sha(target)})
    result = {'frozen_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'selection_rule': 'Pre-existing explicit path mapping in lob_forecasting/experiments/discrimination_finalize.py:16, also pinned by task; no population-restricted scores computed.',
              'frozen_before_restricted_scoring': True, 'restricted_scores_computed': False,
              'selection_caveat': 'BEST_UC YAML/decision call its choice PAIR_0_BASELINE, whereas the finalizer and task identify F_best_UC_42_stage2. Its decompressed predictions/config/history/manifest equal PAIR_5_SHALLOW_INTERACTION_UC_42. Preserve this metadata inconsistency; use the explicit frozen artifact, not a new winner.',
              'corrected_um_evidence': 'Existing selected_configurations.yaml selects UM_V2_DEPTH; per-run manifests checked. Original UM is excluded.',
              'selected': selected, 'missing': missing, 'files': files,
              'predictions_ready': 'true' if len(selected) == 9 and not missing else 'partial' if selected else 'false',
              'checkpoint_payloads': 0}
    (docs / 'prediction_index.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'selected_filesets': len(selected), 'missing': missing, 'bytes_packaged': sum(x['bytes'] for x in files),
                      'index_sha256': sha(docs / 'prediction_index.json')}))


if __name__ == '__main__':
    main()
