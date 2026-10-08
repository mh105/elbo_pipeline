import unittest
from unittest.mock import patch

import mne
import numpy as np

from test_bad_ch_cross_channel import SFREQ, WINDOW, field_raw, step2
from test_ransac_flatline import flatline_result


def native_field(seconds=24):
    raw = field_raw(seconds=seconds)
    raw._data -= raw._data[raw.ch_names.index("Cz")].copy()
    return raw


class TestSmallDetection(unittest.TestCase):
    def test_attenuation_is_flagged_and_amplification_is_not(self):
        raw = native_field()
        raw._data[raw.ch_names.index("E10")] *= 0.1
        raw._data[raw.ch_names.index("E75")] *= 2
        raw._data[raw.ch_names.index("E5")] = np.nan
        raw.info["bads"] = ["E5"]
        auxiliary = mne.io.RawArray(
            np.full((1, raw.n_times), np.nan), mne.create_info(["EOG"], SFREQ, "eog"),
            verbose=False,
        )
        raw.add_channels([auxiliary], force_update_info=True)
        original = raw.copy()
        report, details = step2.bad_ch_by_small(raw, return_details=True)
        self.assertIn("E10", report["small"])
        self.assertNotIn("E75", report["small"])
        self.assertFalse(set(["Cz", "E5", "EOG"]) & set(details["ch_names"]))
        self.assertEqual(raw.info["bads"], ["E5"] + report["small"])
        np.testing.assert_array_equal(raw.get_data(), original.get_data())
        self.assertEqual(raw.annotations, original.annotations)
        self.assertEqual(step2.bad_ch_by_small(original), report)

    def test_small_native_reference_neighbour_is_not_flagged_solely_for_amplitude(self):
        raw = native_field()
        positions = np.array([ch["loc"][:3] for ch in raw.info["chs"]])
        reference = raw.ch_names.index("Cz")
        nearest = np.argsort(np.linalg.norm(positions - positions[reference], axis=1))[1]
        report = step2.bad_ch_by_small(raw)
        self.assertNotIn(raw.ch_names[nearest], report["small"])

    def test_flat_targets_use_actual_mad_including_partial_flat_windows(self):
        raw = native_field(seconds=8)
        row = raw.ch_names.index("E119")
        raw._data[row, WINDOW:2 * WINDOW] = 17e-6
        raw._data[row, 3 * WINDOW + 20:3 * WINDOW + 30] = 17e-6
        flatlines = flatline_result(raw, {
            "E119": [[WINDOW, 2 * WINDOW], [3 * WINDOW + 20, 3 * WINDOW + 30]],
        })
        original = raw.copy()
        report, details = step2.bad_ch_by_small(raw, flatlines=flatlines, return_details=True)
        index = details["ch_names"].index("E119")
        observed = details["observed_mad_uv"][index]
        self.assertEqual(observed[1], 0)
        self.assertGreater(observed[3], 0)
        for window in range(4):
            signal = original._data[row, window * WINDOW:(window + 1) * WINDOW]
            expected = np.median(np.abs(signal - np.median(signal))) * 1e6
            self.assertAlmostEqual(observed[window], expected)
        self.assertTrue((details["predicted_mad_uv"][index, [1, 3]] > 0).all())
        self.assertTrue(details["retained"].all())
        self.assertEqual(details["ratio"][index, 1], 0)
        self.assertTrue(np.isfinite(details["robust_z"][index, 1]))
        self.assertLess(details["robust_z"][index, 1], -5)
        self.assertEqual(details["violates"][index, 1], 1)
        self.assertEqual(details["n_scorable"][index], 4)
        self.assertNotIn("E119", report["small"])
        np.testing.assert_array_equal(details["flagged_data_uv"]["E119"],
                                      original._data[row, WINDOW:2 * WINDOW][None] * 1e6)

    def test_predictions_match_mne_and_repeated_donor_sets_reuse_splines(self):
        raw = native_field(seconds=8)
        raw.info["bads"] = ["E5"]
        flatlines = flatline_result(raw, {
            "E75": [[WINDOW + 20, WINDOW + 30], [3 * WINDOW + 20, 3 * WINDOW + 30]],
        })
        original = raw.copy()
        with patch.object(step2, "_spline_weights", wraps=step2._spline_weights) as spline:
            with patch.object(step2, "solve", wraps=step2.solve) as solve:
                _, details = step2.bad_ch_by_small(raw, flatlines=flatlines, return_details=True)
        keys = {(tuple(call.args[1]), tuple(call.args[2])) for call in spline.call_args_list}
        self.assertEqual(solve.call_count, len(keys))
        self.assertLess(solve.call_count, len(details["ch_names"]) * 4)
        np.testing.assert_array_equal(details["configuration_index"], [0, 1, 0, 1])
        for donors, targets in keys:
            self.assertIn(raw.ch_names.index("Cz"), donors)
            self.assertNotIn(raw.ch_names.index("E5"), donors)
            self.assertFalse(set(donors) & set(targets))
        for target in ("E10", "E75", "E119"):
            index = details["ch_names"].index(target)
            for window in (0, 1):
                expected = original.copy()
                expected.info["bads"] = list(dict.fromkeys(
                    ["E5", target] + (["E75"] if window == 1 else [])
                ))
                expected.interpolate_bads(reset_bads=False, verbose=False)
                signal = expected.get_data(
                    picks=[target], start=window * WINDOW, stop=(window + 1) * WINDOW,
                )[0]
                expected_mad = np.median(np.abs(signal - np.median(signal))) * 1e6
                self.assertAlmostEqual(details["predicted_mad_uv"][index, window],
                                       expected_mad, places=7)

    def test_one_usable_ratio_leaves_the_window_unscorable(self):
        raw = native_field(seconds=8).pick(["E1", "E2", "Cz"])
        raw._data[0, WINDOW:2 * WINDOW] = 17e-6
        flatlines = flatline_result(raw, {"E1": [[WINDOW, 2 * WINDOW]]})
        _, details = step2.bad_ch_by_small(raw, flatlines=flatlines, return_details=True)
        self.assertEqual(details["ch_names"], ["E1", "E2"])
        self.assertTrue(details["retained"].all())
        self.assertGreater(details["predicted_mad_uv"][0, 1], 0)
        self.assertEqual(details["predicted_mad_uv"][1, 1], 0)
        self.assertEqual(details["ratio"][0, 1], 0)
        self.assertTrue(np.isnan(details["ratio"][1, 1]))
        for name in ("robust_z", "violates"):
            self.assertTrue(np.isnan(details[name][:, 1]).all())
        np.testing.assert_array_equal(details["n_scorable"], [3, 3])

    def test_masks_and_trailing_partial_preserve_scorable_window_support(self):
        raw = native_field(seconds=10.2)
        raw._data[raw.ch_names.index("E10")] *= 0.05
        raw.set_annotations(mne.Annotations([2.1], [0.2], ["BAD_test"]))
        raw._data[:40, 6 * SFREQ:6 * SFREQ + 25] += 800e-6
        original = raw.copy()
        report, details = step2.bad_ch_by_small(raw, return_details=True)
        self.assertEqual(step2.bad_ch_by_small(original.copy()), report)
        np.testing.assert_array_equal(details["retained"], [True, False, True, False, True])
        np.testing.assert_array_equal(details["start_s"], [0, 2, 4, 6, 8])
        np.testing.assert_array_equal(details["stop_s"], [2, 4, 6, 8, 10])
        for name in ("observed_mad_uv", "predicted_mad_uv", "ratio", "robust_z", "violates"):
            self.assertTrue(np.isnan(details[name][:, [1, 3]]).all())
        np.testing.assert_array_equal(details["configuration_index"][[1, 3]], [-1, -1])
        ratios = details["observed_mad_uv"] / details["predicted_mad_uv"]
        np.testing.assert_allclose(details["ratio"], ratios)
        logs = np.log(ratios[:, details["retained"]])
        center = np.median(logs, axis=0)
        scale = 1.4826 * np.median(np.abs(logs - center), axis=0)
        np.testing.assert_allclose(details["robust_z"][:, details["retained"]],
                                   (logs - center) / scale)
        np.testing.assert_array_equal(details["n_scorable"], np.full(len(ratios), 3))
        self.assertFalse(set(["n_pairs", "t_stat", "p_value", "p_value_fdr"]) & set(details))
        for row, name in enumerate(details["ch_names"]):
            windows = np.flatnonzero(details["violates"][row] == 1)
            waveforms = details["flagged_data_uv"][name]
            self.assertEqual(waveforms.shape, (len(windows), WINDOW))
            for waveform, window in zip(waveforms, windows):
                expected = original.get_data(
                    picks=[name], start=window * WINDOW, stop=(window + 1) * WINDOW,
                )[0] * 1e6
                np.testing.assert_array_equal(waveform, expected)
        self.assertEqual(details["flagged_data_uv"]["E10"].shape, (3, WINDOW))
        self.assertEqual(details["flagged_data_uv"]["E75"].shape, (0, WINDOW))

    def test_robust_z_and_occurrence_cutoffs_are_strict(self):
        raw = native_field(seconds=8).pick(["E1", "E2", "E3", "E4", "Cz"])
        scores = np.array([[-5, -5, -5, -5], [-6, 0, 0, 0],
                           [-6, -6, 0, 0], [0, 0, 0, 0]])
        with patch.object(step2, "_robust_channel_z", side_effect=scores.T) as robust:
            report, details = step2.bad_ch_by_small(raw, return_details=True)
        self.assertEqual(report, {"small": ["E3"]})
        np.testing.assert_array_equal(details["robust_z"], scores)
        np.testing.assert_array_equal(details["violates"], scores < -5)
        for window, call in enumerate(robust.call_args_list):
            np.testing.assert_allclose(call.args[0], np.log(details["ratio"][:, window]))
            self.assertEqual(call.kwargs["scale_floor"], 1e-6)

    def test_zero_predictions_are_missing_from_the_occurrence_denominator(self):
        for scale in (0, 1e-12):
            with self.subTest(scale=scale):
                raw = native_field(seconds=8)
                raw._data[:, :WINDOW] *= scale
                raw._data[raw.ch_names.index("E10"), WINDOW:2 * WINDOW] *= 0.05
                report, details = step2.bad_ch_by_small(raw, return_details=True)
                self.assertTrue(details["retained"].all())
                for name in ("ratio", "robust_z", "violates"):
                    self.assertTrue(np.isnan(details[name][:, 0]).all())
                np.testing.assert_array_equal(details["n_scorable"],
                                              np.full(len(details["ch_names"]), 3))
                self.assertIn("E10", report["small"])

    def test_shared_amplitude_changes_and_repeated_windows_preserve_flags(self):
        raw = native_field(seconds=8)
        raw._data[raw.ch_names.index("E10"), :2 * WINDOW] *= 0.05
        report, details = step2.bad_ch_by_small(raw.copy(), return_details=True)
        quiet = raw.copy()
        quiet._data[:, WINDOW:2 * WINDOW] *= 0.05
        quiet._data[:, 3 * WINDOW:4 * WINDOW] *= 0.1
        quiet_report, quiet_details = step2.bad_ch_by_small(quiet, return_details=True)
        self.assertEqual(quiet_report, report)
        for metric in ("ratio", "robust_z"):
            np.testing.assert_allclose(quiet_details[metric], details[metric],
                                       rtol=1e-10, atol=1e-10)
        repeated = mne.io.RawArray(np.tile(raw._data, (1, 3)), raw.info.copy(), verbose=False)
        repeated_report, repeated_details = step2.bad_ch_by_small(repeated, return_details=True)
        self.assertEqual(repeated_report, report)
        np.testing.assert_allclose(repeated_details["robust_z"],
                                   np.tile(details["robust_z"], (1, 3)))

    def test_rounding_does_not_split_an_identical_ratio_baseline(self):
        raw = native_field(seconds=4).pick(["E1", "E2", "E3", "E4", "Cz"])
        observed = np.array([1 - 1e-12, 1, 1, 1]) * 1e-5
        prescribed = iter([observed, np.full(4, 1e-5)] * 2)
        original_mad = step2._mad

        def window_mad(values, axis=0):
            return next(prescribed) if values.ndim == 2 else original_mad(values, axis)

        with patch.object(step2, "_mad", side_effect=window_mad):
            report, details = step2.bad_ch_by_small(raw, return_details=True)
        self.assertEqual(report, {"small": []})
        self.assertLess(np.max(np.abs(details["robust_z"])), 1e-5)

    def test_native_reference_is_required_and_another_zero_channel_can_be_named(self):
        for reference_channel in (None, "Cz"):
            raw = field_raw(seconds=4)
            original = raw.copy()
            with self.subTest(reference_channel=reference_channel):
                with self.assertRaisesRegex(ValueError, "exactly one|not entirely zero"):
                    step2.bad_ch_by_small(raw, reference_channel=reference_channel)
                self.assertEqual(raw.info["bads"], [])
                np.testing.assert_array_equal(raw.get_data(), original.get_data())
                self.assertEqual(raw.annotations, original.annotations)
        raw = native_field()
        raw._data[raw.ch_names.index("E119")] = 0
        with self.assertRaisesRegex(ValueError, "exactly one"):
            step2.bad_ch_by_small(raw)
        report = step2.bad_ch_by_small(raw, reference_channel="Cz")
        self.assertIn("E119", report["small"])

    def test_named_native_reference_is_kept_as_a_donor_when_bad(self):
        raw = native_field(seconds=4)
        raw.info["bads"] = ["Cz"]
        with patch.object(step2, "_spline_weights", wraps=step2._spline_weights) as spline:
            _, details = step2.bad_ch_by_small(raw, reference_channel="Cz", return_details=True)
        self.assertNotIn("Cz", details["ch_names"])
        for call in spline.call_args_list:
            self.assertIn(raw.ch_names.index("Cz"), call.args[1])

    def test_native_small_flags_are_initial_bads_for_robust_reference(self):
        raw = native_field()
        raw._data[raw.ch_names.index("E10")] *= 0.05
        native = raw.copy()
        small = step2.bad_ch_by_small(raw)
        self.assertIn("E10", small["small"])
        report, details = step2.robust_average_reference_inplace(raw, return_details=True)
        self.assertEqual(details["initial_bads"], small["small"])
        self.assertNotIn("small", report)
        for iteration in details["iterations"]:
            self.assertNotIn("small", iteration)
            self.assertNotIn("E10", iteration["std"]["ch_names"])
        expected = native.copy()
        flagged = set(small["small"]).union(*report.values())
        expected.info["bads"] = [name for name in raw.ch_names if name in flagged]
        expected.interpolate_bads(reset_bads=True, verbose=False)
        expected.set_eeg_reference("average", projection=False, verbose=False)
        np.testing.assert_allclose(raw.get_data(), expected.get_data(), rtol=0, atol=1e-14)
        self.assertEqual(raw.info["bads"], [])

    def test_one_window_is_allowed_and_missing_windows_or_invalid_cutoffs_raise(self):
        _, details = step2.bad_ch_by_small(native_field(seconds=2), return_details=True)
        np.testing.assert_array_equal(details["n_scorable"],
                                      np.ones(len(details["ch_names"]), dtype=int))
        raw = native_field(seconds=4)
        raw.set_annotations(mne.Annotations([0], [4], ["BAD_test"]))
        with self.assertRaisesRegex(ValueError, "No complete unmasked windows"):
            step2.bad_ch_by_small(raw)
        self.assertEqual(raw.info["bads"], [])
        for parameter, cutoffs in (("robust_z_cutoff", (0, -0.1, np.inf, np.nan)),
                                   ("occurrence_cutoff", (-0.1, 1.1, np.inf, np.nan))):
            for cutoff in cutoffs:
                with self.subTest(parameter=parameter, cutoff=cutoff):
                    with self.assertRaisesRegex(ValueError, parameter):
                        step2.bad_ch_by_small(native_field(seconds=4), **{parameter: cutoff})


if __name__ == "__main__":
    unittest.main()
