import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import step2_bad_ch_time_detection as step2


NAMES = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7",
         "C3", "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4",
         "P8", "PO3", "PO4", "O1", "Oz", "O2"]


def make_native_raw(noisy=None, seconds=40, sfreq=250):
    """Smooth scalp fields plus sensor noise in the native Cz reference, plus an EOG channel.

    The fields give neighbouring channels the shared signal that bad_ch_by_correlation
    expects. Each field averages to zero across channels and every channel gets
    the same field power, so the average-referenced variance is uniform,
    as with independent noise. The noisy channel gets ten times more independent noise.
    """
    rng = np.random.default_rng(11)
    n_times = int(seconds * sfreq)
    info = mne.create_info(NAMES + ["EOG"], sfreq, ["eeg"] * len(NAMES) + ["eog"])
    raw = mne.io.RawArray(np.zeros((len(NAMES) + 1, n_times)), info, verbose=False)
    raw.set_montage(mne.channels.make_standard_montage("colin27_1020"), verbose=False)
    patterns = np.array([ch["loc"][:3] for ch in raw.info["chs"][:len(NAMES)]])
    for _ in range(50):  # Alternate zero-mean fields and equal channel power until both hold.
        patterns -= patterns.mean(axis=0)
        patterns /= np.linalg.norm(patterns, axis=1, keepdims=True)
    potentials = patterns @ rng.normal(scale=10e-6, size=(3, n_times))
    potentials += rng.normal(scale=1e-6, size=potentials.shape)
    if noisy is not None:
        potentials[NAMES.index(noisy)] += rng.normal(scale=100e-6, size=n_times)
    raw._data[:len(NAMES)] = potentials - potentials[NAMES.index("Cz")]
    raw._data[-1] = rng.normal(scale=50e-6, size=n_times)
    return raw


def interpolated_average(raw, bads):
    """Independent reference result: interpolate bads, then subtract the all-EEG mean."""
    expected = raw.copy()
    expected.info["bads"] = list(bads)
    if bads:
        expected.interpolate_bads(reset_bads=True, verbose=False)
    eeg = mne.pick_types(expected.info, eeg=True, exclude=[])
    mean = expected._data[eeg].mean(axis=0)
    expected._data[eeg] -= mean
    return expected, mean


def scheduled_detector(schedule, criterion="HFnoise"):
    """Stand-in for a detector that flags schedule[pass] (last entry repeats)."""
    calls = []

    def detector(raw, **kwargs):
        names = schedule[min(len(calls), len(schedule) - 1)]
        calls.append(names)
        raw.info["bads"] = list(dict.fromkeys(raw.info["bads"] + names))
        result = {criterion: list(names)}
        return (result, {}) if kwargs.get("return_details", False) else result
    return detector, calls


class TestRobustAverageReference(unittest.TestCase):
    def setUp(self):
        plot_report = patch.object(step2, "plot_bad_ch_report")
        self.plot_report = plot_report.start()
        self.addCleanup(plot_report.stop)

    def test_interpolates_all_bads_and_subtracts_the_robust_mean(self):
        raw = make_native_raw(noisy="O1")
        raw._data[NAMES.index("T7"), 100:200] = np.nan  # Interpolation never reads bads.
        raw.info["bads"] = ["T7", "EOG"]
        native = raw.copy()

        def check_uninterpolated(plotted_raw, report):
            self.assertIs(plotted_raw, raw)
            self.assertEqual(plotted_raw.info["bads"], ["T7", "EOG"])
            np.testing.assert_array_equal(plotted_raw.get_data(), native.get_data())
            self.assertEqual(report["std"], ["O1"])

        self.plot_report.side_effect = check_uninterpolated
        report, details = step2.robust_average_reference_inplace(raw, return_details=True)
        self.plot_report.assert_called_once_with(raw, report)
        reference = details["final_reference"]

        self.assertEqual(report, {"voltage": [], "HFnoise": [], "pop": [], "std": ["O1"],
                                  "BBnoise": [], "correlation": []})
        self.assertEqual(details["initial_bads"], ["T7"])
        self.assertEqual(set().union(*report.values()), {"O1"})
        self.assertEqual(raw.info["bads"], ["EOG"])
        expected, mean = interpolated_average(native, ["T7", "O1"])
        eeg = mne.pick_types(raw.info, eeg=True, exclude=[])
        np.testing.assert_allclose(raw._data[eeg], expected._data[eeg], rtol=0, atol=1e-15)
        np.testing.assert_allclose(reference, mean, rtol=0, atol=1e-15)
        np.testing.assert_allclose(raw._data[eeg].mean(axis=0), 0, rtol=0, atol=1e-18)
        np.testing.assert_array_equal(raw._data[NAMES.index("Cz")], -reference)
        np.testing.assert_array_equal(raw.get_data(picks="EOG"), native.get_data(picks="EOG"))
        self.assertIsNone(step2._reference_channel(raw, "average"))

    def test_details_describe_each_pass_without_changing_the_result(self):
        raw = make_native_raw(noisy="O1")
        raw.info["bads"] = ["T7", "EOG"]
        expected = raw.copy()
        expected_report = step2.robust_average_reference_inplace(expected)
        report, details = step2.robust_average_reference_inplace(raw, return_details=True)
        self.assertIsInstance(expected_report, dict)
        self.assertEqual(report, expected_report)
        self.assertNotIn("initial", report)
        self.assertNotIn("initial_bads", report)
        self.assertEqual(details["initial_bads"], ["T7"])
        self.assertIsInstance(details, dict)
        self.assertEqual(set(details), {"iterations", "final_reference", "initial_bads"})
        np.testing.assert_array_equal(details["final_reference"], -expected._data[NAMES.index("Cz")])
        self.assertEqual(details["final_reference"].shape, (raw.n_times,))
        self.assertEqual(details["final_reference"].dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(raw.get_data(), expected.get_data())
        iterations = details["iterations"]
        self.assertEqual([d["interpolated"] for d in iterations], [["T7"], ["T7", "O1"]])
        self.assertEqual([d["flagged"] for d in iterations], [["O1"], ["O1"]])
        for criterion in ("voltage", "HFnoise", "pop", "std", "BBnoise",
                          "correlation"):
            self.assertIn(criterion, iterations[-1])
        # std flags O1, so the detectors that run after it in the pass no longer evaluate it.
        self.assertIn("O1", iterations[-1]["std"]["ch_names"])
        for criterion in ("BBnoise", "correlation"):
            self.assertNotIn("O1", iterations[-1][criterion]["ch_names"])
        voltage = iterations[-1]["voltage"]
        self.assertEqual(len(voltage["exceed_fraction"]), len(voltage["ch_names"]))
        self.assertTrue((voltage["exceed_fraction"] <= 0.25).all())

    def test_small_is_not_run_in_any_pass(self):
        with patch.object(step2, "bad_ch_by_small", side_effect=AssertionError(
            "Smallness must only be checked before robust average referencing."
        )) as small:
            report, details = step2.robust_average_reference_inplace(
                make_native_raw(noisy="O1"), return_details=True,
            )
        small.assert_not_called()
        self.assertEqual(len(details["iterations"]), 2)
        self.assertNotIn("small", report)
        for iteration in details["iterations"]:
            self.assertNotIn("small", iteration)

    def test_stops_when_a_pass_repeats_and_drops_unflagged_channels(self):
        # F4 is flagged once, then not again: no union, so it is not interpolated.
        for schedule, passes, final in (([[]], 1, []),
                                        ([["F3", "F4"], ["F3"]], 3, ["F3"]),
                                        ([["F3"], ["F3", "Pz"]], 3, ["F3", "Pz"])):
            with self.subTest(schedule=schedule):
                raw = make_native_raw()
                native = raw.copy()
                detector, calls = scheduled_detector(schedule)
                with patch.object(step2, "bad_ch_by_HFnoise", detector):
                    report, details = step2.robust_average_reference_inplace(raw, return_details=True)
                reference = details["final_reference"]
                self.assertEqual(len(calls), passes)
                self.assertEqual(details["initial_bads"], [])
                self.assertEqual(report["HFnoise"], final)
                self.assertEqual(set().union(*report.values()), set(final))
                expected, mean = interpolated_average(native, final)
                np.testing.assert_allclose(raw.get_data(), expected.get_data(),
                                           rtol=0, atol=1e-15)
                np.testing.assert_allclose(reference, mean, rtol=0, atol=1e-15)

    def test_cycles_mark_every_channel_flagged_in_the_cycle_as_bad(self):
        # schedule, passes run, channels treated as bad, flipping channels, repeated pass
        for schedule, passes, final, flipping, repeated in (
            ([["F3", "F4"], ["F3"], ["F3", "F4"]], 3, ["F3", "F4"], ["F4"], 2),
            ([["F3"], ["F4"], ["F3"]], 3, ["F3", "F4"], ["F3", "F4"], 2),
            ([["F3"], []], 2, ["F3"], ["F3"], 1),  # Back to the first mean's empty set.
        ):
            with self.subTest(schedule=schedule):
                raw = make_native_raw()
                native = raw.copy()
                detector, calls = scheduled_detector(schedule)
                with patch.object(step2, "bad_ch_by_HFnoise", detector), self.assertWarnsRegex(
                    UserWarning, rf"pass {passes + 1} would repeat pass {repeated}, with "
                                 rf"{re.escape(repr(flipping))} flipping.*as bad: "
                                 rf"{re.escape(repr(final))}"
                ):
                    report, details = step2.robust_average_reference_inplace(raw, return_details=True)
                reference = details["final_reference"]
                self.assertEqual(len(calls), passes)
                self.assertEqual(details["initial_bads"], [])
                self.assertEqual(report["HFnoise"], final)
                self.assertEqual(set().union(*report.values()), set(final))
                expected, mean = interpolated_average(native, final)
                np.testing.assert_allclose(raw.get_data(), expected.get_data(),
                                           rtol=0, atol=1e-15)
                np.testing.assert_allclose(reference, mean, rtol=0, atol=1e-15)

    def test_cycle_report_merges_only_the_cycling_passes(self):
        # Pass 1 (mean of []) flags F3 by std. Pass 2 (mean of [F3]) flags F3 and
        # F4 by HFnoise. Pass 3 (mean of [F3, F4]) flags F3 by HFnoise, the set pass 2's
        # mean used, so passes 2 and 3 cycle. Pass 1's std label is outside the cycle.
        std, _ = scheduled_detector([["F3"], [], []], criterion="std")
        hf, calls = scheduled_detector([[], ["F3", "F4"], ["F3"]])
        with patch.object(step2, "bad_ch_by_std", std), patch.object(
            step2, "bad_ch_by_HFnoise", hf
        ), self.assertWarnsRegex(UserWarning, r"pass 4 would repeat pass 2, with \['F4'\]"):
            report = step2.robust_average_reference_inplace(make_native_raw())
        self.assertEqual(len(calls), 3)
        self.assertEqual(report["HFnoise"], ["F3", "F4"])
        self.assertEqual(report["std"], [])

    def test_raises_without_changes_if_passes_neither_converge_nor_cycle(self):
        raw = make_native_raw()
        original = raw.get_data().copy()
        detector, calls = scheduled_detector([["F3"], ["F4"], ["Fz"], ["Pz"]])
        with patch.object(step2, "bad_ch_by_HFnoise", detector), self.assertRaisesRegex(
            RuntimeError, r"did not converge in 3 passes.*\[\['F3'\], \['F4'\], \['Fz'\]\]"
        ):
            step2.robust_average_reference_inplace(raw, max_iterations=3)
        self.assertEqual(len(calls), 3)
        self.assertEqual(raw.info["bads"], [])
        np.testing.assert_array_equal(raw.get_data(), original)

    def test_ransac_is_not_run_in_any_pass(self):
        with patch.object(step2, "bad_ch_by_RANSAC", side_effect=AssertionError(
            "RANSAC must not run during robust average referencing."
        )) as ransac:
            report, details = step2.robust_average_reference_inplace(
                make_native_raw(noisy="O1"), return_details=True,
            )
        ransac.assert_not_called()
        self.assertEqual(len(details["iterations"]), 2)
        self.assertNotIn("RANSAC", report)
        for iteration in details["iterations"]:
            self.assertNotIn("RANSAC", iteration)

    def test_input_requirements(self):
        for max_iterations in (0, 1.5):
            with self.subTest(max_iterations=max_iterations), self.assertRaises(ValueError):
                step2.robust_average_reference_inplace(make_native_raw(),
                                                       max_iterations=max_iterations)
        raw = make_native_raw()
        raw._data[0, 5] = np.nan
        with self.assertRaisesRegex(ValueError, "bad_ch_by_nan"):
            step2.robust_average_reference_inplace(raw)


if __name__ == "__main__":
    unittest.main()
