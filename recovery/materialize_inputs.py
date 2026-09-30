"""Restore ONLY manifest-listed, already-published aggregate inputs into a new directory.

Python 3.9+, standard library only. No conversion, inference, training, package changes,
checkpoint recovery, or Git ref/config changes. A promisor clone may fetch the requested
blob on demand through its existing authorized remote. Fetch the documented input
commit with blob filtering before invocation; never use an unfiltered fallback.
"""
import argparse
import csv
import hashlib
from pathlib import Path
import re
import subprocess


def verified_file(path, row):
    if not path.is_file() or path.stat().st_size != int(row['bytes']):
        return False
    digest = hashlib.sha256()
    blob = hashlib.sha1(('blob ' + str(int(row['bytes'])) + '\0').encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
            blob.update(block)
    return digest.hexdigest() == row['sha256'] and blob.hexdigest() == row['source_git_blob']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('input_manifest.csv'))
    parser.add_argument('--verify-only', action='store_true', help='Read/check an existing destination without changing any files.')
    args = parser.parse_args()
    with args.manifest.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 54 or len({r['source_path'] for r in rows}) != 54:
        raise SystemExit('Expected exactly 54 distinct approved input paths.')
    for row in rows:
        path = row['source_path']
        allowed = re.fullmatch(r'data/raw/KRXaggData/KR[0-9]+_t\.xlsx', path) or re.fullmatch(r'data/processed/KR[0-9]+_t\.csv', path)
        if not allowed or not re.fullmatch(r'[0-9a-f]{40}', row['source_commit']) or not re.fullmatch(r'[0-9a-f]{40}', row['source_git_blob']) or not re.fullmatch(r'[0-9a-f]{64}', row['sha256']):
            raise SystemExit('Invalid manifest path or object identity.')
        if not 0 < int(row['bytes']) < 20_000_000:
            raise SystemExit('Unexpected input size; review the manifest.')
        obj = subprocess.run(['git', '-C', str(args.repository), 'rev-parse', row['source_commit'] + ':' + path], text=True, capture_output=True)
        if obj.returncode or obj.stdout.strip() != row['source_git_blob']:
            raise SystemExit('Input commit/tree unavailable or mismatched. Fetch the documented published input commit with blob filtering; no fallback performed.')
    if args.verify_only:
        if not all(verified_file(args.destination / row['source_path'], row) for row in rows):
            raise SystemExit('Existing input checksum/size mismatch or missing file.')
        print('Verified all 54 existing inputs against their published Git blobs and SHA-256; no files changed.')
        return
    # Refuse to overwrite any existing destination, including an empty directory.
    args.destination.mkdir(parents=True, exist_ok=False)
    total = 0
    for row in rows:
        target = args.destination / row['source_path']
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + '.part')
        with partial.open('xb') as stream:
            result = subprocess.run(['git', '-C', str(args.repository), 'cat-file', 'blob', row['source_git_blob']], stdout=stream, stderr=subprocess.PIPE)
        if result.returncode:
            raise SystemExit('Required published blob unavailable; partial destination retained for inspection. No unrestricted fetch attempted.')
        if not verified_file(partial, row):
            raise SystemExit('Input checksum/size mismatch; partial destination retained for inspection.')
        partial.rename(target)
        total += int(row['bytes'])
    print('Recovered and verified {} existing input files ({} bytes). No conversion performed.'.format(len(rows), total))


if __name__ == '__main__':
    main()
