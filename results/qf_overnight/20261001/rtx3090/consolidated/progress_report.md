# Matched all-server analysis

Status: partial

Source result commits: {"a5000": "e2dc9ba7db90614633970c4a4410d566dd7a89e8", "rtx3090": "cd6b7594d14793efe3e76a677887712426c776fa"}

Validated selected metric rows: 208; complete paired rows: 40; incomplete seed comparisons: 320.

No missing models/assets/seeds are inner-joined away or recorded as zero. Historical replay is separate. Table exports must match pinned per-date key/label digests, population counts and scientific hashes before pairing.

Issues: ["rtx3090: independent_analysis/metrics.csv missing columns ['config_hash', 'core_source_digest', 'input_hash', 'mask_hash', 'panel', 'protocol_hash', 'row_key_label_digest', 'selected', 'source_digest', 'source_sha', 'split_hash']"]

Daily sums and key/label digests originate in owner-validated saved-prediction exports. No probability ensemble; no inference or selection here. Intervals retrospective/conditional on selection. Missing results are not zero.
