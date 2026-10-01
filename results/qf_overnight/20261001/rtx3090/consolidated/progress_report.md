# Matched all-server analysis

Status: partial

Source result commits: {"a5000": "3a355b8fd9eb60f742207db29478bb5085e7449d", "rtx3090": "90898e0bc4a2c59b0111f19ef0acfb1f048b9049"}

Validated selected metric rows: 488; complete paired rows: 456; incomplete seed comparisons: 16.

No missing models/assets/seeds are inner-joined away or recorded as zero. Historical replay is separate. Table exports must match pinned per-date key/label digests, population counts and scientific hashes before pairing.

Issues: ["rtx3090: independent_analysis/metrics.csv missing columns ['config_hash', 'core_source_digest', 'input_hash', 'mask_hash', 'panel', 'protocol_hash', 'row_key_label_digest', 'selected', 'source_digest', 'source_sha', 'split_hash']"]

Daily sums and key/label digests originate in owner-validated saved-prediction exports. No probability ensemble; no inference or selection here. Intervals retrospective/conditional on selection. Missing results are not zero.
