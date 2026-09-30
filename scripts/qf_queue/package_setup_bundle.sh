#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); STAMP=$(date +%Y%m%d_%H%M%S); OUT=/home/hosung/UPLOAD_ME_qf_queue_setup_${STAMP}.zip; TMP=$(mktemp -d)
cd "$ROOT"; git diff HEAD^..HEAD > "$TMP/latest.patch"; git log --oneline > "$TMP/commits.txt"; cp -a lob_forecasting configs/qf experiments/qf scripts/qf_queue "$TMP/"; cp QUEUE_SETUP_REPORT.md queue_*.csv resource_estimate.csv model_*.csv smoke_validation_report.md "$TMP/" 2>/dev/null || true
(cd "$TMP" && find . -type f -print0 | sort -z | xargs -0 sha256sum > SHA256_MANIFEST.txt)
python -c 'import pathlib,sys,zipfile;r=pathlib.Path(sys.argv[1]);z=zipfile.ZipFile(sys.argv[2],"w",zipfile.ZIP_DEFLATED);[z.write(p,p.relative_to(r)) for p in r.rglob("*") if p.is_file()];z.close()' "$TMP" "$OUT"; echo "$OUT"
