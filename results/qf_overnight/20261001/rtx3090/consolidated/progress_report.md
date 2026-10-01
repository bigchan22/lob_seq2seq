# Matched all-server analysis

Status: partial

Source result commits: {"a5000": "54a10ea64ce815db430e6fd987d9ae9add5e5d12", "rtx3090": "bfee169fa6f3301b9fd4522046cd3389b25ce4cf"}

Validated selected metric rows: 344; complete paired rows: 296; incomplete seed comparisons: 136.

No missing models/assets/seeds are inner-joined away or recorded as zero. Historical replay is separate. Table exports must match pinned per-date key/label digests, population counts and scientific hashes before pairing.

Issues: ["rtx3090: independent_analysis/metrics.csv missing columns ['config_hash', 'core_source_digest', 'input_hash', 'mask_hash', 'panel', 'protocol_hash', 'row_key_label_digest', 'selected', 'source_digest', 'source_sha', 'split_hash']"]

Daily sums and key/label digests originate in owner-validated saved-prediction exports. No probability ensemble; no inference or selection here. Intervals retrospective/conditional on selection. Missing results are not zero.
