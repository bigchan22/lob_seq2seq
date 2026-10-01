# Matched all-server analysis

Status: partial

Source result commits: {"a5000": "09b744a319e63ece4f1a7f13b00e341a17c2f53d", "rtx3090": "c3f3f201b74fc1ae69866f4f81f92f4b6c360bc8"}

Validated selected metric rows: 408; complete paired rows: 384; incomplete seed comparisons: 72.

No missing models/assets/seeds are inner-joined away or recorded as zero. Historical replay is separate. Table exports must match pinned per-date key/label digests, population counts and scientific hashes before pairing.

Issues: ["rtx3090: independent_analysis/metrics.csv missing columns ['config_hash', 'core_source_digest', 'input_hash', 'mask_hash', 'panel', 'protocol_hash', 'row_key_label_digest', 'selected', 'source_digest', 'source_sha', 'split_hash']"]

Daily sums and key/label digests originate in owner-validated saved-prediction exports. No probability ensemble; no inference or selection here. Intervals retrospective/conditional on selection. Missing results are not zero.
