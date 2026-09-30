#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); STAMP=$(date +%Y%m%d_%H%M%S); OUT=/home/hosung/UPLOAD_ME_qf_pilot_results_${STAMP}.zip
python - "$ROOT/artifacts" "$OUT" <<'PY'
import pathlib,sys,zipfile
r=pathlib.Path(sys.argv[1]);z=zipfile.ZipFile(sys.argv[2],'w',zipfile.ZIP_DEFLATED)
for p in r.rglob('*'):
 if p.is_file() and p.suffix not in {'.pt','.pth'} and ('predictions' not in p.parts or p.stat().st_size<100_000_000):z.write(p,p.relative_to(r))
z.close()
PY
echo "$OUT"
