# Matched all-server analysis

Status: partial

Source result commits: {"a5000": "99ce58961e401b8ef77f6cb712ecc620f983b1ae", "rtx3090": "8001ee16c7f8cf5ec1051f47bdf9391bbb3ef38a"}

Validated selected metric rows: 480; complete paired rows: 448; incomplete seed comparisons: 24.

No missing models/assets/seeds are inner-joined away or recorded as zero. Historical replay is separate. Table exports must match pinned per-date key/label digests, population counts and scientific hashes before pairing.

Issues: ["rtx3090: independent_analysis/metrics.csv missing columns ['config_hash', 'core_source_digest', 'input_hash', 'mask_hash', 'panel', 'protocol_hash', 'row_key_label_digest', 'selected', 'source_digest', 'source_sha', 'split_hash']"]

Daily sums and key/label digests originate in owner-validated saved-prediction exports. No probability ensemble; no inference or selection here. Intervals retrospective/conditional on selection. Missing results are not zero.
