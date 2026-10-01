# Matched all-server analysis

Status: partial

Source result commits: {"a5000": "54a10ea64ce815db430e6fd987d9ae9add5e5d12", "rtx3090": "b217af5e45e64a580e963c9b67ecc6bc8dbbab51"}

Validated selected metric rows: 280; complete paired rows: 208; incomplete seed comparisons: 200.

No missing models/assets/seeds are inner-joined away or recorded as zero. Historical replay is separate. Table exports must match pinned per-date key/label digests, population counts and scientific hashes before pairing.

Issues: ["rtx3090: independent_analysis/metrics.csv missing columns ['config_hash', 'core_source_digest', 'input_hash', 'mask_hash', 'panel', 'protocol_hash', 'row_key_label_digest', 'selected', 'source_digest', 'source_sha', 'split_hash']"]

Daily sums and key/label digests originate in owner-validated saved-prediction exports. No probability ensemble; no inference or selection here. Intervals retrospective/conditional on selection. Missing results are not zero.
