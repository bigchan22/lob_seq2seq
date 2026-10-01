# Forecast-row schema

UTF-8 CSV inside deterministic gzip; empty source_time means zero_fill. Read identifiers and all date/time fields as strings, with `keep_default_na=False` if preserving empty strings. Flags are integers 0/1. Provider observation/release semantics remain unknown.

| column | type | meaning |
| --- | --- | --- |
| asset_id | string | String equal to input filename stem, including _t; no issuer-name inference. |
| date | string | Common date serialized YYYY-MM-DD; Asia/Seoul local-clock convention. |
| origin_time | string | Forecast origin HH:MM:SS; fixed grid 09:00:00 through 15:10:00. |
| endpoint_time | string | Same-date target endpoint HH:MM:SS, exactly origin plus ten minutes. |
| split | string | train, validation, or development_holdout; whole common dates, chronological. |
| legacy_label | integer | 0=Down, 1=Flat, 2=Up using the original float32 +/-0.0001 rule. |
| no_cross_day_target | integer | 1 iff origin/endpoint are on the same date; all rows are 1. |
| exact_ten_minute_target | integer | 1 iff endpoint is exactly origin plus ten minutes; all rows are 1. |
| scheduled_session_regime | string | pre_20160801_extension or from_20160801_extension; documented schedule era, not a per-day session certification. |
| exceptional_session_status | string | Always unresolved; no inferred halt, holiday, auction, or exception-day flag. |
| origin_cache_row_exists | integer | At origin: original cache contains this exact date/time row, before reindexing/filling. |
| origin_workbook_row_exists | integer | At origin: supplied workbook contains this exact normalized date/time row. |
| origin_workbook_cache_status | string | At origin: full-row status matched_exact, matched_tolerance, conflict, cache_only, workbook_only, or absent_in_both; numeric tolerance atol=rtol=1e-12. |
| origin_file_value_conflict | integer | At origin: 1 for conflict/cache_only/workbook_only, otherwise 0. |
| origin_cache_ask1_present | integer | At origin: ask1 original cache cell is nonmissing before filling (infinity would be present but nonfinite). |
| origin_cache_ask1_finite | integer | At origin: ask1 original cache cell is finite before filling. |
| origin_cache_ask1_finite_positive | integer | At origin: ask1 original cache cell is finite and >0 before filling. |
| origin_cache_ask1_nonpositive | integer | At origin: ask1 original cache cell is finite and <=0 before filling. |
| origin_ask1_fill_method | string | At origin: ask1 original (present), forward_fill (earlier same-day on-grid value), or zero_fill (no earlier present grid value). Zero-valued original cells stay original. |
| origin_ask1_source_time | string | At origin: ask1 supplying same-day grid time HH:MM:SS, or empty for zero_fill; origin fill source never exceeds origin. |
| origin_workbook_ask1_present | integer | At origin: ask1 workbook cell nonmissing; not an exchange observation flag. |
| origin_cache_bid1_present | integer | At origin: bid1 original cache cell is nonmissing before filling (infinity would be present but nonfinite). |
| origin_cache_bid1_finite | integer | At origin: bid1 original cache cell is finite before filling. |
| origin_cache_bid1_finite_positive | integer | At origin: bid1 original cache cell is finite and >0 before filling. |
| origin_cache_bid1_nonpositive | integer | At origin: bid1 original cache cell is finite and <=0 before filling. |
| origin_bid1_fill_method | string | At origin: bid1 original (present), forward_fill (earlier same-day on-grid value), or zero_fill (no earlier present grid value). Zero-valued original cells stay original. |
| origin_bid1_source_time | string | At origin: bid1 supplying same-day grid time HH:MM:SS, or empty for zero_fill; origin fill source never exceeds origin. |
| origin_workbook_bid1_present | integer | At origin: bid1 workbook cell nonmissing; not an exchange observation flag. |
| origin_crossed | integer | At origin: both original cache quotes finite and bid1>ask1; nonpositive values are separately flagged. |
| origin_locked_positive | integer | At origin: both original cache quotes finite and positive and bid1==ask1. |
| origin_valid_original_cache_quote | integer | At origin: original bid1/ask1 finite, strictly positive, and not crossed; locked positive quotes allowed. |
| endpoint_cache_row_exists | integer | At endpoint: original cache contains this exact date/time row, before reindexing/filling. |
| endpoint_workbook_row_exists | integer | At endpoint: supplied workbook contains this exact normalized date/time row. |
| endpoint_workbook_cache_status | string | At endpoint: full-row status matched_exact, matched_tolerance, conflict, cache_only, workbook_only, or absent_in_both; numeric tolerance atol=rtol=1e-12. |
| endpoint_file_value_conflict | integer | At endpoint: 1 for conflict/cache_only/workbook_only, otherwise 0. |
| endpoint_cache_ask1_present | integer | At endpoint: ask1 original cache cell is nonmissing before filling (infinity would be present but nonfinite). |
| endpoint_cache_ask1_finite | integer | At endpoint: ask1 original cache cell is finite before filling. |
| endpoint_cache_ask1_finite_positive | integer | At endpoint: ask1 original cache cell is finite and >0 before filling. |
| endpoint_cache_ask1_nonpositive | integer | At endpoint: ask1 original cache cell is finite and <=0 before filling. |
| endpoint_ask1_fill_method | string | At endpoint: ask1 original (present), forward_fill (earlier same-day on-grid value), or zero_fill (no earlier present grid value). Zero-valued original cells stay original. |
| endpoint_ask1_source_time | string | At endpoint: ask1 supplying same-day grid time HH:MM:SS, or empty for zero_fill; origin fill source never exceeds origin. |
| endpoint_workbook_ask1_present | integer | At endpoint: ask1 workbook cell nonmissing; not an exchange observation flag. |
| endpoint_cache_bid1_present | integer | At endpoint: bid1 original cache cell is nonmissing before filling (infinity would be present but nonfinite). |
| endpoint_cache_bid1_finite | integer | At endpoint: bid1 original cache cell is finite before filling. |
| endpoint_cache_bid1_finite_positive | integer | At endpoint: bid1 original cache cell is finite and >0 before filling. |
| endpoint_cache_bid1_nonpositive | integer | At endpoint: bid1 original cache cell is finite and <=0 before filling. |
| endpoint_bid1_fill_method | string | At endpoint: bid1 original (present), forward_fill (earlier same-day on-grid value), or zero_fill (no earlier present grid value). Zero-valued original cells stay original. |
| endpoint_bid1_source_time | string | At endpoint: bid1 supplying same-day grid time HH:MM:SS, or empty for zero_fill; origin fill source never exceeds origin. |
| endpoint_workbook_bid1_present | integer | At endpoint: bid1 workbook cell nonmissing; not an exchange observation flag. |
| endpoint_crossed | integer | At endpoint: both original cache quotes finite and bid1>ask1; nonpositive values are separately flagged. |
| endpoint_locked_positive | integer | At endpoint: both original cache quotes finite and positive and bid1==ask1. |
| endpoint_valid_original_cache_quote | integer | At endpoint: original bid1/ask1 finite, strictly positive, and not crossed; locked positive quotes allowed. |
| origin_history_file_conflict | integer | Any workbook/cache full-row conflict at an on-grid time up to and including origin in this asset-day; conservative provenance flag, not a quote-observation flag. |
| workbook_cache_filled_mid_equal_origin | integer | Exact equality of float32 filled mids at origin under independent workbook/cache reconstructions. |
| workbook_cache_filled_mid_equal_endpoint | integer | Exact equality of float32 filled mids at endpoint under independent workbook/cache reconstructions. |
| workbook_cache_label_match | integer | Exact class agreement under independent workbook/cache float32 reconstructions. |
| missing_cache_quote_any_endpoint | integer | Any missing (NaN/absent row) original bid1/ask1 at either endpoint; does not equate zero with missing. |
| historical_missing_cache_endpoint | integer | Earlier audit definition: any absent/nonfinite original bid1/ask1 at either endpoint. Distinct from the positive, uncrossed P1 rule. |
| historical_missing_endpoint_kind | string | none, current, future, both according to historical_missing_cache_endpoint; disjoint categories. |
| equal_filled_mids | integer | Exact equality of filled float32 current/future mids; NOT evidence of provider imputation. |
| historical_missing_equal_filled_flat | integer | Historical missing endpoint AND equal filled mids AND legacy_label=1; overlapping subset of historical missing, not an independent exclusion. |
| candidate_clock | integer | origin>=09:10:00 AND endpoint<=14:40:00, with the verified same-day ten-minute target; origins 09:10–14:30 inclusive. |
| P0_legacy | integer | 1 for every legacy row. |
| P1_valid_cache_endpoints | integer | Both original endpoints have finite positive bid1/ask1 and bid1<=ask1, with same-day ten-minute target. Positive locks retained. Ex-post evaluation eligibility. |
| P2_candidate_clock | integer | P0 restricted only to candidate_clock; independent of quote validity. |
| P3_intersection | integer | P1_valid_cache_endpoints AND P2_candidate_clock. Ex-post evaluation eligibility. |
| peer_origin_available_cache | integer | Only origin_valid_original_cache_quote. Never uses a future endpoint or P1/P3; cache-based availability proxy, not provider release certification. |
| P1_exclusion_reasons | string | none or pipe-separated endpoint-qualified row_absent, ask1/bid1_missing, nonfinite, nonpositive, crossed; multiple reasons may overlap. |
| P2_exclusion_reason | string | none, origin_before_0910, or endpoint_after_1440. |
