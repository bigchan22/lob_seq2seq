# Matched all-server analysis

Status: partial

Source result commits: {"a5000": "094273fcec8a4ee33dfb89a75789effd8e75c61d", "rtx3090": "ec5b1bcc63ad652fa9b35a290bb510ae8cff6fd2"}

Validated selected metric rows: 432; complete paired rows: 400; incomplete seed comparisons: 56.

No missing models/assets/seeds are inner-joined away or recorded as zero. Historical replay is separate. Table exports must match pinned per-date key/label digests, population counts and scientific hashes before pairing.

Issues: ["rtx3090: independent_analysis/metrics.csv missing columns ['config_hash', 'core_source_digest', 'input_hash', 'mask_hash', 'panel', 'protocol_hash', 'row_key_label_digest', 'selected', 'source_digest', 'source_sha', 'split_hash']"]

Daily sums and key/label digests originate in owner-validated saved-prediction exports. No probability ensemble; no inference or selection here. Intervals retrospective/conditional on selection. Missing results are not zero.
