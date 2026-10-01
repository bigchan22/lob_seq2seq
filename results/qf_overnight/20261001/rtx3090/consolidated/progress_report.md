# Matched all-server analysis

Status: partial

Source result commits: {"a5000": "f8af150012a05f43cae2c5fdb5c5b08d6d0cfaec", "rtx3090": "b2e041ff65cd011501bc57d783bd17a96db47351"}

Validated selected metric rows: 488; complete paired rows: 456; incomplete seed comparisons: 16.

No missing models/assets/seeds are inner-joined away or recorded as zero. Historical replay is separate. Table exports must match pinned per-date key/label digests, population counts and scientific hashes before pairing.

Issues: ["rtx3090: independent_analysis/metrics.csv missing columns ['config_hash', 'core_source_digest', 'input_hash', 'mask_hash', 'panel', 'protocol_hash', 'row_key_label_digest', 'selected', 'source_digest', 'source_sha', 'split_hash']"]

Daily sums and key/label digests originate in owner-validated saved-prediction exports. No probability ensemble; no inference or selection here. Intervals retrospective/conditional on selection. Missing results are not zero.
