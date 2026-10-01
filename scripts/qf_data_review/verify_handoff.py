"""Read-only, standard-library verification of the committed task packet."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def digest(path, compressed=False):
    value = hashlib.sha256()
    with (gzip.open(path, 'rb') if compressed else path.open('rb')) as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    args = parser.parse_args(); root = args.repository.resolve()
    manifest_path = root / 'docs/qf_data_review/20261001/handoff.json'
    handoff = json.loads(manifest_path.read_text())
    paths = set()
    for record in handoff['task_files']:
        path = root / record['path']
        path.resolve().relative_to(root)  # reject paths outside the packet
        assert path != manifest_path and record['path'] not in paths
        paths.add(record['path'])
        assert path.is_file() and not path.is_symlink(), record['path']
        assert path.stat().st_size == record['bytes'], record['path']
        assert digest(path) == record['sha256'], record['path']
        if 'decompressed_sha256' in record:
            assert digest(path, True) == record['decompressed_sha256'], record['path']
    for name in ['input_manifest', 'code_manifest']:
        path = root / handoff[name]
        path.resolve().relative_to(root)
        assert digest(path) == handoff[name + '_sha256']
    for prediction in handoff['selected_prediction_index']:
        assert prediction['prediction_path'] in paths
        path = root / prediction['prediction_path']
        assert digest(path) == prediction['prediction_sha256']
        assert digest(path, True) == prediction['decompressed_sha256']
    print(json.dumps({'task_files_verified': len(paths),
                      'selected_prediction_files_verified': len(handoff['selected_prediction_index']),
                      'input_and_code_manifest_hashes_verified': True,
                      'writes': 0, 'training_or_inference_run': False}))


if __name__ == '__main__':
    main()
