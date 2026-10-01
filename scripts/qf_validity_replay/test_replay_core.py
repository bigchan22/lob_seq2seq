"""Targeted synthetic integrity tests; no project data or model execution."""
import unittest

import numpy as np
import pandas as pd

from replay_core import (ContractError, checked_probabilities,
                         deterministic_table_digest, endpoint_populations,
                         labels_from_raw_aligned_prices, match_predictions,
                         moving_block_interval, score_probabilities)


class ReplayIntegrityTests(unittest.TestCase):
    def test_normalization_and_clip_convention(self):
        values = np.array([[0., .3, .7000001], [.1, .8, .1]])
        result, loss, diagnostic = score_probabilities([0, 1], values)
        expected = [-np.log(np.finfo(np.float64).eps), -np.log(.8)]
        np.testing.assert_allclose(loss, expected, rtol=0, atol=1e-14)
        self.assertAlmostEqual(result['nll'], np.mean(expected))
        self.assertGreater(diagnostic['row_sum_error_max'], 0)

    def test_invalid_probabilities_are_not_hidden_by_normalization(self):
        for p in [[[0, 0, 0]], [[-.1, .5, .6]], [[np.nan, .5, .5]],
                  [[.2, .3, .8]], [[0, 0, 1.00001]]]:
            with self.assertRaises(ContractError):
                checked_probabilities(p)

    def test_exact_key_set_and_labels(self):
        expected = pd.DataFrame({'asset_id': ['A', 'B'], 'date': ['D1', 'D1'],
                                 'origin_time': ['09:10:00', '09:10:00'],
                                 'true_class': [0, 2]})
        aligned = match_predictions(expected, expected.iloc[::-1])
        self.assertEqual(aligned.asset_id.tolist(), ['A', 'B'])
        for bad in [expected.iloc[:1], pd.concat([expected, expected.iloc[:1]]),
                    expected.assign(true_class=[0, 1]), expected.assign(asset_id=['A', 'C'])]:
            with self.assertRaises(ContractError):
                match_predictions(expected, bad)
        self.assertEqual(deterministic_table_digest(expected, expected.columns),
                         deterministic_table_digest(expected.iloc[::-1], expected.columns))

    def test_raw_validity_and_clock_boundaries(self):
        # P1 rejects zero, NaN, crossed and cross-day endpoints separately.
        bid = np.array([100., 100., 0., np.nan, 102., 100., 100.])
        ask = np.array([101., 101., 1., 101., 101., 101., 101.])
        origin = np.array([9*3600, 9*3600+600, 10*3600, 10*3600,
                           10*3600, 14*3600+1800, 14*3600+2400])
        masks, _ = endpoint_populations(bid, ask, np.full(7, 100.),
                                        np.full(7, 101.), origin, origin+600,
                                        np.ones(7, dtype=bool))
        np.testing.assert_array_equal(masks['P1'], [1, 1, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(masks['P2'], [0, 1, 1, 1, 1, 1, 0])
        np.testing.assert_array_equal(masks['P3'], [0, 1, 0, 0, 0, 1, 0])
        cross_day, _ = endpoint_populations([100], [101], [100], [101],
                                            [33000], [33600], np.array([False]))
        self.assertFalse(cross_day['P1'][0])
        self.assertFalse(cross_day['P2'][0])

    def test_legacy_float32_labels_and_no_cross_day_fill(self):
        raw = np.array([[np.nan, 10000., 10001., 10003., 10001.],
                        [np.nan, np.nan, 20000., 20000., 20000.]])
        labels, mids = labels_from_raw_aligned_prices(raw, raw)
        # Exactly 1/10000 rounds below the positive strict float64 threshold;
        # +2/10001 is Up, -2/10003 is Down. Initial missing endpoint is Flat.
        np.testing.assert_array_equal(labels[0], [1, 1, 2, 0])
        np.testing.assert_array_equal(mids[1, :2], [0, 0])
        self.assertEqual(mids.dtype, np.float32)

    def test_bootstrap_uses_pooled_rows_and_is_reproducible(self):
        sums = np.array([1., -18., 1., -18., 1., -18.])
        counts = np.array([1, 9, 1, 9, 1, 9])
        out = moving_block_interval(sums, counts, block_length=5,
                                    replicates=10000, seed=20261001)
        self.assertAlmostEqual(out['delta_nll'], -1.7)
        self.assertNotAlmostEqual(out['delta_nll'], np.mean(sums/counts))
        self.assertEqual(out, moving_block_interval(sums, counts, block_length=5,
                                                   replicates=10000, seed=20261001))
        constant = moving_block_interval(-.02*counts, counts, block_length=5,
                                         replicates=10000, seed=20261001)
        self.assertAlmostEqual(constant['ci_lower'], -.02)
        self.assertAlmostEqual(constant['ci_upper'], -.02)


if __name__ == '__main__':
    unittest.main(verbosity=2)
