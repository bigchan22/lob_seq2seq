# Matched all-server analysis

Status: partial

Source result commits: {"a5000": "4dedb9a7be46e1c258d9096cd93ae70df55ed704", "rtx3090": "6e1eea1d844ff25e2e5c6a4df3be45188848b590"}

Validated selected metric rows: 488; complete paired rows: 456; incomplete seed comparisons: 16.

No missing models/assets/seeds are inner-joined away or recorded as zero. Historical replay is separate. Table exports must match pinned per-date key/label digests, population counts and scientific hashes before pairing.

Issues: ["rtx3090: independent_analysis/metrics.csv missing columns ['config_hash', 'core_source_digest', 'input_hash', 'mask_hash', 'panel', 'protocol_hash', 'row_key_label_digest', 'selected', 'source_digest', 'source_sha', 'split_hash']"]

Daily sums and key/label digests originate in owner-validated saved-prediction exports. No probability ensemble; no inference or selection here. Intervals retrospective/conditional on selection. Missing results are not zero.
