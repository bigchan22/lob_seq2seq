"""CPU-only primitives for the August saved-probability replay.

These functions do not discover artifacts, choose models, infer a producer schema,
load checkpoints, or build model inputs. The producer adapter remains blocked
until the exact data-review handoff and frozen artifact index are available.
"""
from __future__ import annotations

import csv
import hashlib
import io

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

KEYS = ("asset_id", "date", "origin_time")
CLASS_ORDER = ("Down", "Flat", "Up")
PROBABILITY_SUM_TOLERANCE = 1e-5
FLOAT64_EPSILON = np.finfo(np.float64).eps


class ContractError(ValueError):
    """A mismatch must be investigated, never repaired by a silent inner join."""


def checked_labels(values):
    y = np.asarray(values)
    if y.ndim != 1 or not np.isin(y, [0, 1, 2]).all():
        raise ContractError("Labels must be a one-dimensional array of 0/1/2.")
    return y.astype(np.int64)


def checked_probabilities(values, sum_tolerance=PROBABILITY_SUM_TOLERANCE):
    """Validate before normalization; ordering must explicitly be Down/Flat/Up."""
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p):
        raise ContractError("Expected a nonempty N x 3 probability array.")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ContractError("Nonfinite or out-of-bounds probabilities.")
    sums = p.sum(axis=1, dtype=np.float64)
    if (sums <= 0).any():
        raise ContractError("Probability row sums must be strictly positive.")
    error = np.abs(sums - 1.0)
    if np.max(error) > sum_tolerance:
        raise ContractError("Large probability row-sum error: %.17g" % error.max())
    diagnostics = {
        "rows": len(p),
        "row_sum_error_max": float(error.max()),
        "row_sum_error_mean": float(error.mean()),
        "row_sum_error_p99": float(np.quantile(error, 0.99)),
        "allowed_sum_error": float(sum_tolerance),
        "minimum_probability": float(p.min()),
        "maximum_probability": float(p.max()),
    }
    return p / sums[:, None], diagnostics


def score_probabilities(labels, probabilities):
    """Equal forecast-row weight, float64 normalization, eps clip, natural log."""
    y = checked_labels(labels)
    p, diagnostics = checked_probabilities(probabilities)
    if len(y) != len(p):
        raise ContractError("Label and probability row counts differ.")
    loss = -np.log(np.maximum(p[np.arange(len(y)), y], FLOAT64_EPSILON))
    predicted = np.argmax(p, axis=1)
    metrics = {
        "rows": len(y),
        "nll": float(loss.mean(dtype=np.float64)),
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, labels=[0, 1, 2],
                                   average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, predicted)),
    }
    return metrics, loss, diagnostics


def _checked_key_index(frame):
    missing = set(KEYS) - set(frame.columns)
    if missing:
        raise ContractError("Missing canonical key columns: " + repr(sorted(missing)))
    keys = frame.loc[:, list(KEYS)]
    if keys.isna().any().any():
        raise ContractError("Null canonical key.")
    # The reviewed producer adapter must supply exact strings. Do not strip,
    # parse/reformat dates, remove ticker suffixes, or infer asset identities here.
    if not all(isinstance(value, str) and value and value == value.strip()
               for value in keys.to_numpy().ravel()):
        raise ContractError("Canonical keys must already be nonempty exact strings.")
    index = pd.MultiIndex.from_frame(keys)
    if index.has_duplicates:
        raise ContractError("Duplicate canonical keys.")
    return index


def match_predictions(expected, predictions, label_column="true_class"):
    """Require exactly the expected row set; align by key, not CSV row order."""
    a = _checked_key_index(expected)
    b = _checked_key_index(predictions)
    absent, extra = a.difference(b), b.difference(a)
    if len(absent) or len(extra):
        raise ContractError("Incomplete support: missing=%d extra=%d" %
                            (len(absent), len(extra)))
    positions = b.get_indexer(a)
    if (positions < 0).any():
        raise ContractError("Unexpected unmatched key after support assertion.")
    aligned = predictions.iloc[positions].reset_index(drop=True)
    y = checked_labels(expected[label_column].to_numpy())
    actual = checked_labels(aligned[label_column].to_numpy())
    if not np.array_equal(y, actual):
        raise ContractError("Labels differ on %d matched rows." % np.count_nonzero(y != actual))
    return aligned


def deterministic_table_digest(frame, columns):
    """Analysis-local digest definition; do not assume the producer uses it.

    Exact strings, sorted by canonical keys; RFC-style CSV quoting, UTF-8,
    comma delimiter, LF newline, header included, no index. Producer digests
    must be verified using their separately documented serialization.
    """
    _checked_key_index(frame)
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(columns)
    for row in frame.sort_values(list(KEYS), kind="mergesort")[list(columns)].itertuples(index=False, name=None):
        writer.writerow(row)
    return hashlib.sha256(out.getvalue().encode("utf-8")).hexdigest()


def labels_from_raw_aligned_prices(bid, ask):
    """Independent legacy label primitive; each row is one already aligned day.

    Missing cache values are forward-filled *within that day*, then zero-filled.
    Midpoint, subtraction and division use float32, as in the committed August
    bake_tensors implementation. Compare float32 results as scalar values to
    Python's +/-0.0001 threshold (cast returns to float64 for this comparison).
    No model/cache conversion or tensor construction takes place.
    """
    bid, ask = np.asarray(bid, dtype=np.float64), np.asarray(ask, dtype=np.float64)
    if bid.ndim != 2 or bid.shape != ask.shape or bid.shape[1] < 2:
        raise ContractError("Expected matching day-by-endpoint price matrices.")
    b = pd.DataFrame(bid).ffill(axis=1).fillna(0).to_numpy(dtype=np.float32)
    a = pd.DataFrame(ask).ffill(axis=1).fillna(0).to_numpy(dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mid = (a + b) / np.float32(2.0)
        returns = ((mid[:, 1:] - mid[:, :-1]) / mid[:, :-1]).astype(np.float64)
    y = np.ones(returns.shape, dtype=np.int64)
    usable_origin = mid[:, :-1] > 0
    y[usable_origin & (returns < -0.0001)] = 0
    y[usable_origin & (returns > +0.0001)] = 2
    return y, mid


def endpoint_populations(bid_origin, ask_origin, bid_endpoint, ask_endpoint,
                         origin_seconds, endpoint_seconds, same_day):
    """Predeclared population formulas, applied only to ORIGINAL cache values.

    Call only after the producer key/date/grid contract is independently checked.
    Seconds are literal wall-clock seconds from midnight, not inferred exchange
    observation/release times. These ex-post masks are not trading rules.
    """
    prices = np.column_stack([bid_origin, ask_origin, bid_endpoint, ask_endpoint]).astype(np.float64)
    n = len(prices)
    origin, endpoint = np.asarray(origin_seconds), np.asarray(endpoint_seconds)
    same_day = np.asarray(same_day)
    if origin.shape != (n,) or endpoint.shape != (n,) or same_day.shape != (n,) or same_day.dtype != bool:
        raise ContractError("Clock/date flags must be explicit aligned vectors; same_day must be boolean.")
    if not np.isfinite(origin).all() or not np.isfinite(endpoint).all():
        raise ContractError("Nonfinite clock values.")
    finite = np.isfinite(prices).all(axis=1)
    positive = (prices > 0).all(axis=1)
    uncrossed = (prices[:, 0] <= prices[:, 1]) & (prices[:, 2] <= prices[:, 3])
    ten_minutes = same_day & ((endpoint - origin) == 600)
    p1 = finite & positive & uncrossed & ten_minutes
    p2 = ten_minutes & (origin >= 9 * 3600 + 10 * 60) & (endpoint <= 14 * 3600 + 40 * 60)
    masks = {"P0": np.ones(n, dtype=bool), "P1": p1, "P2": p2, "P3": p1 & p2}
    # Reasons intentionally overlap. Their sum is not a count of excluded rows.
    reasons = {
        "nonfinite_endpoint_price": ~finite,
        "nonpositive_endpoint_price": (prices <= 0).any(axis=1),
        "crossed_endpoint": (prices[:, 0] > prices[:, 1]) | (prices[:, 2] > prices[:, 3]),
        "not_same_day_ten_minutes": ~ten_minutes,
        "outside_candidate_clock": ~p2,
    }
    return masks, reasons


def moving_block_interval(daily_difference_sums, daily_row_counts, *,
                          block_length=5, replicates=10000, seed=20261001):
    """Paired non-circular moving-date blocks, pooled row weighting.

    Daily sums must already average per-row losses across the SAME matched seed
    set, never seed probabilities. Dates must be consecutive trading dates in
    the declared split, retaining dates with zero selected rows. Keep each date's
    whole asset/model/seed panel together. Draw starts uniformly from 0..D-L,
    concatenate ceil(D/L) blocks and truncate to D. Percentile interval 2.5/97.5.
    """
    sums = np.asarray(daily_difference_sums, dtype=np.float64)
    counts = np.asarray(daily_row_counts, dtype=np.float64)
    if sums.ndim != 1 or sums.shape != counts.shape or not len(sums):
        raise ContractError("Daily sums/counts must be matching nonempty vectors.")
    if not np.isfinite(sums).all() or not np.isfinite(counts).all() or (counts < 0).any() or counts.sum() <= 0:
        raise ContractError("Invalid daily sums/counts.")
    if ((counts == 0) & (sums != 0)).any():
        raise ContractError("Zero-count date has a nonzero loss sum.")
    if not isinstance(block_length, int) or not 1 <= block_length <= len(sums) or replicates < 1:
        raise ContractError("Invalid bootstrap dimensions.")
    rng = np.random.default_rng(seed)
    blocks = (len(sums) + block_length - 1) // block_length
    starts = rng.integers(0, len(sums) - block_length + 1, size=(replicates, blocks))
    indices = (starts[:, :, None] + np.arange(block_length)).reshape(replicates, -1)[:, :len(sums)]
    denominator = counts[indices].sum(axis=1)
    if (denominator <= 0).any():
        raise ContractError("A bootstrap draw has no retained rows; report instead of silently dropping it.")
    samples = sums[indices].sum(axis=1) / denominator
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {
        "delta_nll": float(sums.sum() / counts.sum()),
        "ci_lower": float(lo), "ci_upper": float(hi),
        "replicates": replicates, "block_length": block_length,
        "resampling_seed": seed, "trading_dates": len(sums),
        "method": "non-circular moving-date blocks; pooled row-weighted percentile interval",
    }
