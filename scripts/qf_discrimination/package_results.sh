#!/usr/bin/env bash
set -euo pipefail
: "${QF_DISC_ROOT:?export QF_DISC_ROOT}"
stamp=$(date +%Y%m%d_%H%M%S); out="/home/hosung/UPLOAD_ME_qf_discrimination_results_${stamp}.zip"
(cd "$QF_DISC_ROOT" && find . -type f ! -name '*.pt' ! -name 'test_predictions.csv.gz' ! -name 'placebo_predictions.csv.gz' -print | sort > manifest.files && sha256sum $(cat manifest.files) > SHA256SUMS && zip -q "$out" -@ < manifest.files && zip -q "$out" SHA256SUMS)
sha256sum "$out"; du -h "$out"
