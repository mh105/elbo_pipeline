import unittest

import numpy as np

from test_bad_ch_cross_channel import WINDOW, field_raw, step2
from step2_bad_ch_time_detection import _spline_geometry


def flatline_result(raw, by_name):
    """Original-aligned known segments for testing later average-reference scoring."""
    return {
        "raw_ch_names": list(raw.ch_names),
        "eeg_ch_names": list(raw.ch_names),
        "sfreq": float(raw.info["sfreq"]),
        "first_samp": int(raw.first_samp),
        "n_times": int(raw.n_times),
        "reference_channel": "Cz",
        "ch_names": list(by_name),
        "segments": list(by_name.values()),
        "global_segments": [],
    }


class TestRansacFlatline(unittest.TestCase):
    def test_local_donors_are_substituted_only_in_their_flat_windows(self):
        raw = field_raw(seconds=8)
        raw.info["bads"] = ["E5"]
        detection = flatline_result(raw, {"E119": [[WINDOW + 20, WINDOW + 30]]})
        _, details = step2.bad_ch_by_RANSAC(
            raw, flatlines=detection, n_subsets=8, correlation_cutoff=-1,
            return_details=True,
        )
        np.testing.assert_array_equal(details["configuration_index"], [0, 1, 0, 0])
        baseline, local = details["configurations"]
        np.testing.assert_array_equal(details["subsets"], baseline["subsets"])
        self.assertNotIn("E119", local["ch_names"])
        self.assertFalse(np.isin(["E119", "E5"], local["subsets"]).any())
        self.assertEqual(local["subsets"].shape, (127, 8, 32))
        for target, subsets in zip(local["ch_names"], local["subsets"]):
            before = baseline["subsets"][baseline["ch_names"].index(target)]
            for old, new in zip(before, subsets):
                self.assertNotIn(target, new)
                self.assertEqual(len(set(new)), 32)
                self.assertTrue(set(old) - {"E119"} <= set(new))
        row = details["ch_names"].index("E119")
        self.assertTrue(np.isnan(details["noise_gain_by_window"][row, 1]))
        np.testing.assert_array_equal(details["noise_gain_by_window"][:, 0],
                                      details["noise_gain"])

    def test_flat_windows_are_failures_with_the_full_retained_denominator(self):
        for method in ("threshold", "robust_z"):
            for flat_windows in ([1], [0, 2]):
                with self.subTest(method=method, flat_windows=flat_windows):
                    raw = field_raw(seconds=8)
                    row = raw.ch_names.index("E119")
                    segments = []
                    for index in flat_windows:
                        start, stop = index * WINDOW, (index + 1) * WINDOW
                        raw._data[row, start:stop] = 3e-6
                        segments.append([start, stop])
                    detection = flatline_result(raw, {"E119": segments})
                    report, details = step2.bad_ch_by_RANSAC(
                        raw, method=method, flatlines=detection, n_subsets=8,
                        correlation_cutoff=-1, robust_z_cutoff=1e9, return_details=True,
                    )
                    self.assertEqual(report, {"RANSAC": [] if len(flat_windows) == 1
                                              else ["E119"]})
                    self.assertTrue(details["retained"].all())
                    np.testing.assert_array_equal(details["violates"][row],
                                                  np.isin(np.arange(4), flat_windows))
                    self.assertTrue(np.isnan(details["r"][row, flat_windows]).all())
                    self.assertTrue(np.isnan(details["robust_z"][row, flat_windows]).all())
                    pooled = details["r"][np.isfinite(details["r"])]
                    scale = max(1.4826 * np.median(np.abs(pooled - np.median(pooled))),
                                np.finfo(float).eps)
                    np.testing.assert_allclose(details["robust_z"],
                                               (np.median(pooled) - details["r"]) / scale)
                    for field in ("flagged_data_uv", "flagged_filtered_uv",
                                  "flagged_predicted_uv"):
                        self.assertEqual(details[field]["E119"].shape,
                                         (len(flat_windows), WINDOW))
                    self.assertTrue(np.isnan(details["flagged_predicted_uv"]["E119"]).all())

    def test_too_few_local_donors_raise_without_shrinking_the_fixed_subset(self):
        raw = field_raw(seconds=8)
        raw.info["bads"] = raw.ch_names[:96]
        detection = flatline_result(raw, {"E119": [[WINDOW + 10, WINDOW + 20]]})
        with self.assertRaisesRegex(ValueError, "at least 33 eligible EEG channels.*found 32"):
            step2.bad_ch_by_RANSAC(raw, flatlines=detection, n_subsets=3)

    def test_independent_orderings_and_weights_are_reused_across_calls(self):
        raw = field_raw(seconds=8)
        detection = flatline_result(raw, {
            "E119": [[WINDOW + 10, WINDOW + 20], [3 * WINDOW + 10, 3 * WINDOW + 20]],
            "E75": [[2 * WINDOW + 10, 2 * WINDOW + 20]],
        })
        geometry = _spline_geometry(raw)
        _, first = step2.bad_ch_by_RANSAC(
            raw.copy(), flatlines=detection, n_subsets=8, correlation_cutoff=-1,
            return_details=True, _geometry=geometry,
        )
        cached_weights = len(geometry["weights"])
        _, second = step2.bad_ch_by_RANSAC(
            raw.copy(), flatlines=detection, n_subsets=8, correlation_cutoff=-1,
            return_details=True, _geometry=geometry,
        )
        self.assertEqual(len(geometry["ransac_orderings"]), 1)
        self.assertEqual(len(geometry["weights"]), cached_weights)
        np.testing.assert_array_equal(first["configuration_index"], [0, 1, 2, 1])
        np.testing.assert_array_equal(first["r"], second["r"])
        for before, after in zip(first["configurations"], second["configurations"]):
            np.testing.assert_array_equal(before["subsets"], after["subsets"])

    def test_float32_prediction_has_a_float64_usual_median_and_scores(self):
        rng = np.random.default_rng(12)
        window = rng.normal(scale=1e-6, size=(4, 31))
        for n_subsets in (5, 6, 50):
            with self.subTest(n_subsets=n_subsets):
                weights = rng.normal(size=(n_subsets, 4, 4))
                predicted32 = np.sort(weights.astype(np.float32)
                                      @ window.astype(np.float32), axis=0)
                expected = (predicted32[(n_subsets - 1) // 2].astype(float)
                            + predicted32[n_subsets // 2]) / 2
                # Column t * n_subsets + k holds target t's k-th subset weights.
                columns = weights.transpose(2, 1, 0).reshape(4, -1)
                actual = step2._ransac_median(columns, window, n_subsets)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, np.dtype(float))
                predicted64 = np.sort(weights @ window, axis=0)
                median64 = (predicted64[(n_subsets - 1) // 2]
                            + predicted64[n_subsets // 2]) / 2
                score32 = step2._row_pearson(window, actual)
                score64 = step2._row_pearson(window, median64)
                self.assertEqual(score32.dtype, np.dtype(float))
                np.testing.assert_allclose(score32, score64, atol=2e-7, rtol=0)


if __name__ == "__main__":
    unittest.main()
