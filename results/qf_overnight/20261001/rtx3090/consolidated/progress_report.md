# Matched all-server analysis

Status: partial

Source result commits: {"a5000": "dd84fdcbab6e6ff1faed18039418a82c1a553c1c", "rtx3090": "15439f7c48c5ae28ec809857a6453c37e1b13d56"}

Validated selected metric rows: 488; complete paired rows: 456; incomplete seed comparisons: 16.

No missing models/assets/seeds are inner-joined away or recorded as zero. Historical replay is separate. Table exports must match pinned per-date key/label digests, population counts and scientific hashes before pairing.

Issues: ["rtx3090: independent_analysis/metrics.csv missing columns ['config_hash', 'core_source_digest', 'input_hash', 'mask_hash', 'panel', 'protocol_hash', 'row_key_label_digest', 'selected', 'source_digest', 'source_sha', 'split_hash']"]

Daily sums and key/label digests originate in owner-validated saved-prediction exports. No probability ensemble; no inference or selection here. Intervals retrospective/conditional on selection. Missing results are not zero.
