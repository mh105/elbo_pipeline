import sys
import unittest
from pathlib import Path

import mne
import numpy as np


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import step2_bad_ch_time_detection as step2


MONTAGE = mne.channels.make_standard_montage("GSN-HydroCel-129")
SFREQ = 250
WINDOW = 2 * SFREQ  # Default 2-s windows, in samples.


def field_raw(seconds=60, seed=0):
    """Average-referenced EEG: eight smooth scalp fields shared by all channels plus sensor noise."""
    info = mne.create_info(MONTAGE.ch_names, SFREQ, "eeg")
    n_times = int(seconds * SFREQ)
    raw = mne.io.RawArray(np.zeros((len(MONTAGE.ch_names), n_times)), info, verbose=False)
    raw.set_montage(MONTAGE, verbose=False)
    rng = np.random.default_rng(seed)
    positions = np.array([ch["loc"][:3] for ch in raw.info["chs"]])
    directions = positions - positions.mean(axis=0)
    x, y, z = (directions / np.linalg.norm(directions, axis=1, keepdims=True)).T
    patterns = np.column_stack([x, y, z, x * y, x * z, y * z, x ** 2 - y ** 2, 3 * z ** 2 - 1])
    data = patterns @ rng.normal(scale=10e-6, size=(patterns.shape[1], n_times))
    data += rng.normal(scale=5e-6, size=data.shape)
    raw._data[:] = data - data.mean(axis=0)
    return raw


def replace_windows(raw, name, windows, seed=1):
    """Replace one channel by independent noise in the given 2-s windows, then re-reference."""
    rng = np.random.default_rng(seed)
    row = raw.ch_names.index(name)
    for window in windows:
        raw._data[row, window * WINDOW:(window + 1) * WINDOW] = rng.normal(scale=25e-6, size=WINDOW)
    raw._data -= raw._data.mean(axis=0)
    return raw


def swap(raw, first, second):
    """Exchange two channels' signals, as if their electrodes had been misplaced."""
    rows = [raw.ch_names.index(first), raw.ch_names.index(second)]
    raw._data[rows] = raw._data[rows[::-1]]
    return raw


def faulty_raw():
    """E10 lost in 12 of 30 windows, E75 in 2, and E20/E80 swapped."""
    raw = replace_windows(field_raw(), "E10", range(0, 24, 2))
    return swap(replace_windows(raw, "E75", [5, 20], seed=2), "E20", "E80")


class TestBandLimiting(unittest.TestCase):
    def test_reach_bounds_the_spread_of_both_filters(self):
        impulse = np.zeros((1, 40 * SFREQ))
        impulse[0, 20 * SFREQ] = 1
        filtered, reach = step2._band_limit(impulse, SFREQ)
        # MNE's 1 Hz high-pass (825 taps) applied once, the low-pass (165 taps) forward and backward.
        self.assertEqual(reach, 825 // 2 + 165 - 1)
        support = np.flatnonzero(np.abs(filtered[0]) > 1e-12 * np.abs(filtered).max())
        self.assertGreaterEqual(support.min(), 20 * SFREQ - reach)
        self.assertLessEqual(support.max(), 20 * SFREQ + reach)

    def test_widened_voltage_mask_omits_windows_the_filters_reach(self):
        raw = field_raw()
        raw._data[:40, 30 * SFREQ:30 * SFREQ + 25] += 800e-6  # 31% of channels for 0.1 s at 30 s
        raw._data -= raw._data.mean(axis=0)
        reach = (825 // 2 + 165 - 1) / SFREQ
        for detector in (step2.bad_ch_by_correlation, step2.bad_ch_by_RANSAC):
            with self.subTest(detector=detector.__name__):
                _, details = detector(raw.copy(), return_details=True)
                starts = details["start_s"]
                expected = (details["stop_s"] <= 30 - reach) | (starts >= 30.1 + reach)
                np.testing.assert_array_equal(details["retained"], expected)
                self.assertEqual(int((~details["retained"]).sum()), 4)


class TestCorrelation(unittest.TestCase):
    def test_flags_frequent_signal_loss_but_not_misplaced_channels(self):
        self.assertEqual(step2.bad_ch_by_correlation(field_raw()), {"correlation": []})
        raw = faulty_raw()
        original = raw.copy()
        # E10 lost its signal in 12 of 30 windows, above the 25% cutoff; E75's 2
        # windows are below it. The swapped channels still correlate with their
        # true neighbours, wherever those are.
        self.assertEqual(step2.bad_ch_by_correlation(raw), {"correlation": ["E10"]})
        self.assertEqual(raw.info["bads"], ["E10"])
        np.testing.assert_array_equal(raw.get_data(), original.get_data())
        self.assertEqual(raw.annotations, original.annotations)
        # PREP's 0.4 / 1% cutoffs also catch E75's two windows.
        self.assertEqual(step2.bad_ch_by_correlation(faulty_raw(), correlation_cutoff=0.4,
                                                     occurrence_cutoff=0.01),
                         {"correlation": ["E10", "E75"]})

    def test_score_is_a_quantile_of_signed_r_with_the_other_channels(self):
        raw = field_raw()
        raw._data[raw.ch_names.index("E31")] *= -1  # Inverted: its |r| would be unchanged.
        raw.info["bads"] = ["E5"]
        _, details = step2.bad_ch_by_correlation(raw, return_details=True)
        self.assertNotIn("E5", details["ch_names"])
        picks = [raw.ch_names.index(name) for name in details["ch_names"]]
        filtered, _ = step2._band_limit(raw.get_data(picks=picks), SFREQ)
        r = np.corrcoef(filtered[:, 3 * WINDOW:4 * WINDOW])
        row = details["ch_names"].index("E31")
        others = np.delete(r[row], row)
        np.testing.assert_allclose(details["score"][row, 3], np.quantile(others, 0.98),
                                   rtol=1e-12)
        self.assertLess(details["score"][row, 3], np.quantile(np.abs(others), 0.98) - 0.1)

    def test_constant_window_raises_instead_of_being_handled(self):
        raw = field_raw()
        raw._data[raw.ch_names.index("E10"), 5 * WINDOW:6 * WINDOW] = 3e-6
        for detector in (step2.bad_ch_by_correlation, step2.bad_ch_by_RANSAC):
            with self.subTest(detector=detector.__name__), self.assertRaisesRegex(
                ValueError, r"The input of \['E10'\] is constant in the window starting at 10 s"
            ):
                detector(raw.copy())


class TestRANSAC(unittest.TestCase):
    def test_flags_persistently_unpredictable_and_misplaced_channels_only(self):
        for method in ("threshold", "robust_z"):
            with self.subTest(method=method):
                self.assertEqual(step2.bad_ch_by_RANSAC(field_raw(), method=method),
                                 {"RANSAC": []})
                # E75's 2 bad windows of 30 stay below the 25% cutoff.
                self.assertEqual(step2.bad_ch_by_RANSAC(faulty_raw(), method=method),
                                 {"RANSAC": ["E10", "E20", "E80"]})
                # An inverted channel can still find partners for the correlation
                # detector among channels anticorrelated with its neighbours.
                inverted = field_raw()
                inverted._data[inverted.ch_names.index("E31")] *= -1
                self.assertEqual(step2.bad_ch_by_RANSAC(inverted, method=method),
                                 {"RANSAC": ["E31"]})

    def test_subsets_have_a_fixed_size_exclude_target_and_bads_and_are_stable(self):
        raw = field_raw(seconds=10)
        raw.info["bads"] = ["E5"]
        _, details = step2.bad_ch_by_RANSAC(raw.copy(), return_details=True)
        subsets = details["subsets"]
        self.assertEqual(subsets.shape, (128, 50, 32))  # round(0.25 * 129) = 32
        for name, rows in zip(details["ch_names"], subsets):
            self.assertFalse(np.isin([name, "E5"], rows).any())
            self.assertTrue(all(len(set(subset)) == 32 for subset in rows))
        _, repeated = step2.bad_ch_by_RANSAC(raw.copy(), return_details=True)
        np.testing.assert_array_equal(repeated["subsets"], subsets)
        np.testing.assert_array_equal(repeated["r"], details["r"])
        # A newly bad channel is replaced by the next eligible one; the rest stay.
        raw.info["bads"] = ["E5", "E6"]
        _, fewer = step2.bad_ch_by_RANSAC(raw.copy(), return_details=True)
        kept = np.array(details["ch_names"]) != "E6"
        for before, after in zip(subsets[kept], fewer["subsets"]):
            for old, new in zip(before, after):
                self.assertNotIn("E6", new)
                self.assertTrue(set(old) - {"E6"} <= set(new))

    def test_weights_reproduce_mne_interpolate_bads(self):
        raw = field_raw(seconds=4)
        picks = mne.pick_types(raw.info, eeg=True)
        subsets, weights = step2._ransac_predictors(raw, picks, 3, 32, 435656)
        for row, k in ((0, 0), (64, 1), (128, 2)):
            with self.subTest(channel=raw.ch_names[row], subset=k):
                subset = [raw.ch_names[picks[i]] for i in subsets[row, k]]
                expected = raw.copy()
                expected.info["bads"] = [name for name in raw.ch_names if name not in subset]
                expected.interpolate_bads(reset_bads=False, verbose=False)
                # MNE orders the subset by channel index, so round-off differs slightly.
                np.testing.assert_allclose(weights[row, k] @ raw.get_data(picks=subset),
                                           expected.get_data(picks=[row])[0],
                                           rtol=1e-8, atol=1e-14)

    def test_robust_z_pools_r_over_channels_and_retained_windows(self):
        raw = faulty_raw()
        _, threshold = step2.bad_ch_by_RANSAC(raw.copy(), return_details=True)
        _, robust = step2.bad_ch_by_RANSAC(raw.copy(), method="robust_z", return_details=True)
        np.testing.assert_array_equal(robust["r"], threshold["r"])
        pooled = threshold["r"][:, threshold["retained"]]
        scale = 1.4826 * np.median(np.abs(pooled - np.median(pooled)))
        np.testing.assert_allclose(robust["robust_z"], (np.median(pooled) - robust["r"]) / scale)
        np.testing.assert_array_equal(threshold["violates"], threshold["r"] < 0.5)
        np.testing.assert_array_equal(robust["violates"], robust["robust_z"] > 5)

    def test_details_store_input_band_limited_and_predicted_waveforms(self):
        raw = faulty_raw()
        original = raw.copy()
        _, details = step2.bad_ch_by_RANSAC(raw, return_details=True)
        self.assertEqual(details["noise_gain"].shape, (129,))
        self.assertTrue((details["noise_gain"] > 0).all())
        filtered, _ = step2._band_limit(original.get_data(), SFREQ)
        row = details["ch_names"].index("E10")
        windows = np.flatnonzero(details["violates"][row] == 1)
        self.assertGreater(len(windows), 7)
        for field in ("flagged_data_uv", "flagged_filtered_uv", "flagged_predicted_uv"):
            self.assertEqual(details[field]["E10"].shape, (len(windows), WINDOW))
        for waveforms, source in ((details["flagged_data_uv"]["E10"], original.get_data()),
                                  (details["flagged_filtered_uv"]["E10"], filtered)):
            for waveform, window in zip(waveforms, windows):
                np.testing.assert_allclose(
                    waveform, source[row, window * WINDOW:(window + 1) * WINDOW] * 1e6,
                    rtol=1e-12, atol=1e-12,
                )
        for waveform, filtered_waveform, window in zip(
            details["flagged_predicted_uv"]["E10"], details["flagged_filtered_uv"]["E10"], windows
        ):
            r = np.corrcoef(waveform, filtered_waveform)[0, 1]
            self.assertAlmostEqual(r, details["r"][row, window], places=6)


class TestRequirements(unittest.TestCase):
    def test_native_reference_is_rejected_before_anything_else(self):
        raw = field_raw()
        raw._data -= raw._data[raw.ch_names.index("Cz")]  # Cz as the native reference
        for detector in (step2.bad_ch_by_correlation, step2.bad_ch_by_RANSAC):
            with self.subTest(detector=detector.__name__), self.assertRaisesRegex(
                ValueError, r"\['Cz'\] are entirely zero"
            ):
                detector(raw.copy())

    def test_parameters_positions_and_predictor_count(self):
        raw = field_raw(seconds=10)
        for kwargs in ({"correlation_cutoff": 1.5}, {"quantile": 2}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                step2.bad_ch_by_correlation(raw.copy(), **kwargs)
        for kwargs in ({"method": "cosine"}, {"correlation_cutoff": -2}, {"robust_z_cutoff": 0},
                       {"n_subsets": 0}, {"subset_fraction": 0}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                step2.bad_ch_by_RANSAC(raw.copy(), **kwargs)
        slow = raw.copy().resample(100, verbose=False)
        with self.assertRaisesRegex(ValueError, "above 100 Hz"):
            step2.bad_ch_by_correlation(slow)
        unplaced = mne.io.RawArray(raw.get_data(), mne.create_info(raw.ch_names, SFREQ, "eeg"),
                                   verbose=False)
        with self.assertRaisesRegex(ValueError, "sensor positions"):
            step2.bad_ch_by_RANSAC(unplaced)
        crowded = raw.copy()
        crowded.info["bads"] = crowded.ch_names[:97]  # 32 left: one short of a target plus 32
        with self.assertRaisesRegex(ValueError, "at least 33 non-bad EEG channels; found 32"):
            step2.bad_ch_by_RANSAC(crowded)


if __name__ == "__main__":
    unittest.main()
