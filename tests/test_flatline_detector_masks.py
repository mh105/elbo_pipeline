import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import step2_bad_ch_time_detection as step2
from step2_bad_ch_time_detection import _flatline_configurations
from test_bad_ch_local_signals import set_sphere_positions


SFREQ = 250
WINDOW = 2 * SFREQ


def native_raw(n_windows=4, n_nonreference=12, first_samp=0, tail=0, seed=17):
    rng = np.random.default_rng(seed)
    n_times = n_windows * WINDOW + tail
    data = rng.normal(size=(n_nonreference, n_times))
    data *= np.linspace(8, 12, n_nonreference)[:, None] * 1e-6
    names = [f"E{i}" for i in range(n_nonreference)] + ["Cz"]
    data = np.vstack([data, np.zeros(n_times)])
    return set_sphere_positions(mne.io.RawArray(
        data, mne.create_info(names, SFREQ, "eeg"), first_samp=first_samp, verbose=False,
    ))


def flatlines_for(raw, intervals):
    """Flatline metadata for raw with prescribed segments and no global
    dropouts; tests that need dropouts call _mask_segment_by_flatline."""
    with patch.object(step2.mne.filter, "filter_data", side_effect=lambda data, *a, **k: data.copy()):
        result = step2.detect_flatline_inplace(raw.copy(), reference_channel="Cz")
    result["ch_names"] = list(intervals)
    result["segments_exact"] = list(intervals.values())
    result["segments_bridged"] = list(intervals.values())
    result["segments_on_subtypes"] = {
        "segments_on_raw": list(intervals.values()),
        "segments_on_highpass": [[] for _ in intervals],
        "segments_on_ramp": [[] for _ in intervals],
    }
    result["global_segments_exact"] = []
    return result


def average(raw):
    raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw


DETECTORS = (
    ("std", step2.bad_ch_by_std, "robust_z"),
    ("HFnoise", step2.bad_ch_by_HFnoise, "robust_z"),
    ("BBnoise", step2.bad_ch_by_BBnoise, "robust_z"),
    ("correlation", step2.bad_ch_by_correlation, "score"),
)


class TestPublicFlatlineStages(unittest.TestCase):
    def test_detect_reference_exemption_and_unpadded_channel_bridging(self):
        raw = native_raw(n_nonreference=7)
        raw._data[0, 34:44] = 3e-6
        raw._data[1, 600:609] = 3e-6  # 9 samples: too short to be flat.
        raw.info["bads"] = ["E0"]  # Finite bads are still detected and bridged.
        raw.set_annotations(mne.Annotations([1], [0], ["event"]))
        before = raw.copy()
        with patch.object(step2.mne.filter, "filter_data", side_effect=lambda data, *a, **k: data.copy()):
            result = step2.detect_flatline_inplace(raw)
        self.assertEqual(result["ch_names"], ["E0"])
        self.assertEqual(result["segments_on_subtypes"]["segments_on_raw"], [[[34, 44]]])
        self.assertEqual(result["segments_exact"], [[[34, 44]]])
        self.assertEqual(result["segments_bridged"], [[[34, 44]]])
        self.assertEqual(result["reference_channel"], "Cz")
        self.assertEqual(len(result["eeg_ch_names"]), 8)
        x, original = raw.get_data(), before.get_data()
        line = original[0, 33] + (original[0, 44] - original[0, 33]) * np.arange(1, 11) / 11
        np.testing.assert_allclose(x[0, 34:44], line, rtol=0, atol=1e-18)
        x[0, 34:44] = original[0, 34:44]
        np.testing.assert_array_equal(x, original)
        self.assertEqual(raw.info["bads"], before.info["bads"])
        self.assertEqual(raw.annotations, before.annotations)

    def test_global_annotations_use_exact_spans_and_crop_time_base(self):
        for meas_date in (None, datetime(2026, 10, 5, tzinfo=timezone.utc)):
            with self.subTest(meas_date=meas_date):
                raw = native_raw(n_nonreference=7, first_samp=500)
                raw.set_meas_date(meas_date)
                raw.crop(tmin=0.4)
                flatlines = flatlines_for(raw, {name: [[150, 275]] for name in ("E0", "E1", "E2")})
                raw.info["bads"] = ["E1", "E4", "E5"]
                before = raw.get_data().copy()
                self.assertIsNone(step2._mask_segment_by_flatline(raw, flatlines))
                mask = raw.annotations.description == "BAD_by_flatline"
                np.testing.assert_allclose(raw.annotations.onset[mask], [raw.first_time + 0.6])
                np.testing.assert_allclose(raw.annotations.duration[mask], [0.5])
                self.assertEqual(raw.annotations.orig_time, meas_date)
                np.testing.assert_array_equal(raw.get_data(), before)
                self.assertEqual(raw.info["bads"], ["E1", "E4", "E5"])
                report, details = step2.bad_ch_by_flatline(raw, flatlines=flatlines, return_details=True)
                self.assertEqual(report, {"flatline": []})
                np.testing.assert_array_equal(details["retained"], [False, True, True])

    def test_bad_flatline_any_overlap_strict_cutoff_and_partial_tail(self):
        raw = native_raw(tail=100)
        flatlines = flatlines_for(raw, {
            "E0": [[40, 50], [540, 550]], "E1": [[1040, 1050]], "E2": [[2040, 2050]],
        })
        report, details = step2.bad_ch_by_flatline(raw, flatlines=flatlines, return_details=True)
        self.assertEqual(report, {"flatline": ["E0"]})
        row = details["ch_names"].index("E1")
        np.testing.assert_array_equal(details["flat"][row], [0, 0, 1, 0])
        self.assertNotIn("E2", raw.info["bads"])

    def test_flatline_reference_checks_share_one_zero_scan(self):
        raw = native_raw()
        raw._data[0] = 0  # A dead electrode alongside the named Cz reference.
        flatlines = flatlines_for(raw, {"E0": [[0, raw.n_times]]})
        with patch.object(step2, "_zero_eeg_channels", wraps=step2._zero_eeg_channels) as zeros:
            report = step2.bad_ch_by_flatline(raw, reference_channel="Cz", flatlines=flatlines)
        self.assertEqual(report, {"flatline": ["E0"]})
        uncached = [call for call in zeros.call_args_list
                    if call.kwargs.get("known_zeros", call.args[2] if len(call.args) > 2 else None)
                    is None]
        self.assertEqual(len(uncached), 1)

    def test_flatline_denominator_excludes_the_temporary_voltage_mask(self):
        raw = native_raw()  # 12 channels besides Cz, four 2-s windows.
        raw._data[0, WINDOW:2 * WINDOW] = 3e-6  # E0 flat in window 1.
        raw._data[1:5, 100:105] += 600e-6  # 4 of 12 channels exceed 500 uV in window 0.
        flatlines = step2.detect_flatline_inplace(raw)
        annotations = raw.annotations.copy()
        report, details = step2.bad_ch_by_flatline(raw, flatlines=flatlines,
                                                   return_details=True)
        np.testing.assert_array_equal(details["retained"], [False, True, True, True])
        self.assertEqual(report, {"flatline": ["E0"]})  # 1 of 3 windows, not 1 of 4.
        self.assertEqual(raw.annotations, annotations)

    def test_flatline_mask_accepts_another_all_zero_channel_with_a_named_reference(self):
        raw = native_raw()
        raw._data[3] = 0  # A dead electrode besides the Cz reference.
        flatlines = step2.detect_flatline_inplace(raw, "Cz")
        self.assertEqual(step2.bad_ch_by_flatline(raw, reference_channel="Cz",
                                                  flatlines=flatlines),
                         {"flatline": ["E3"]})

    def test_scoring_consumers_enforce_their_own_window_requirements(self):
        everything = {name: [[0, 4 * WINDOW]] for name in ("E0", "E1", "E2")}
        # Every 2-s window overlaps a dropout, but 1600 of 2000 samples are not flat.
        every_window = {name: [[w * WINDOW + 200, w * WINDOW + 300] for w in range(4)]
                        for name in ("E0", "E1", "E2")}
        for spans in (everything, every_window):
            with self.subTest(spans=spans):
                raw = native_raw(n_nonreference=7)
                flatlines = flatlines_for(raw, spans)
                step2._mask_segment_by_flatline(raw, flatlines)
                with self.assertRaisesRegex(ValueError, "No complete unmasked windows"):
                    step2.bad_ch_by_flatline(raw, reference_channel="Cz", flatlines=flatlines)
        # A 0.5-s window still fits at either edge of the joined annotation.
        raw = native_raw(n_nonreference=7)
        flatlines = flatlines_for(raw, every_window)
        step2._mask_segment_by_flatline(raw, flatlines)
        _, details = step2.bad_ch_by_flatline(raw, reference_channel="Cz", flatlines=flatlines,
                                             window_length=0.5, return_details=True)
        self.assertEqual(int(details["retained"].sum()), 2)

    def test_detection_records_spans_without_a_scoring_window_parameter(self):
        raw = native_raw(n_nonreference=7)
        for row in range(3):
            for w in range(4):
                raw._data[row, w * WINDOW + 200:w * WINDOW + 300] = 3e-6
        flatlines = step2.detect_flatline_inplace(raw)
        self.assertEqual(len(flatlines["global_segments_exact"]), 4)
        self.assertEqual(len(raw.annotations), 7)  # Four events and three inserted gaps.
        with self.assertRaisesRegex(ValueError, "No complete unmasked windows"):
            step2.bad_ch_by_flatline(raw, reference_channel="Cz", flatlines=flatlines)

    def test_masks_share_good_eeg_pool_excluding_native_reference_and_auxiliary_bads(self):
        for reference_is_bad in (False, True):
            with self.subTest(reference_is_bad=reference_is_bad):
                raw = native_raw(n_nonreference=11)
                raw.set_channel_types({"E9": "eog", "E10": "misc"})
                raw.info["bads"] = ["E0", "E1", "E9"] + (["Cz"] if reference_is_bad else [])
                for row in (0, 1, 2, 3, 9, 10):
                    raw._data[row, 20:40] = 600e-6
                for row in (0, 1, 9, 10):
                    raw._data[row, 60:80] = 600e-6
                flatlines = flatlines_for(raw, {
                    "E0": [[20, 40], [60, 80]], "E1": [[20, 40], [60, 80]],
                    "E2": [[20, 40]], "E3": [[20, 40]],
                })
                voltage = raw.copy()
                original = raw.copy()
                step2._mask_segment_by_flatline(raw, flatlines)  # Cz, from detection.
                step2.mask_segment_by_voltage(voltage, window_length=10 / SFREQ)
                self.assertEqual(flatlines["global_ch_names"], [f"E{i}" for i in range(2, 9)])
                self.assertEqual(flatlines["global_segments_exact"], [[20, 40]])  # 2/7 > 25%.
                np.testing.assert_array_equal(raw.annotations.onset, voltage.annotations.onset)
                np.testing.assert_array_equal(raw.annotations.duration, voltage.annotations.duration)
                np.testing.assert_allclose(raw.annotations.onset, [20 / SFREQ])
                np.testing.assert_allclose(raw.annotations.duration, [20 / SFREQ])
                self.assertEqual(raw.info["bads"], original.info["bads"])
                np.testing.assert_array_equal(raw.get_data(), original.get_data())

    def test_mask_cutoff_is_strict_and_excludes_existing_bad_flat_channels(self):
        raw = native_raw(n_nonreference=8)
        flatlines = flatlines_for(raw, {"E0": [[20, 40]], "E1": [[20, 40]]})
        raw._data[:2, 20:40] = 600e-6
        voltage = raw.copy()
        step2._mask_segment_by_flatline(raw, flatlines)
        step2.mask_segment_by_voltage(voltage, window_length=10 / SFREQ)
        self.assertEqual(len(raw.annotations), 0)  # 2/8 is exactly 25%.
        self.assertEqual(len(voltage.annotations), 0)
        raw.info["bads"] = ["E6", "E7"]
        step2._mask_segment_by_flatline(raw, flatlines)
        self.assertEqual(flatlines["global_segments_exact"], [[20, 40]])  # 2/6 > 25%.
        only_bad = native_raw(n_nonreference=8)
        only_bad.info["bads"] = ["E0", "E1", "E2", "E3"]
        flatlines = flatlines_for(only_bad, {name: [[0, only_bad.n_times]]
                                            for name in only_bad.info["bads"]})
        step2._mask_segment_by_flatline(only_bad, flatlines)
        self.assertEqual(len(only_bad.annotations), 0)
        self.assertEqual(flatlines["global_segments_exact"], [])

    def test_detection_and_its_mask_require_the_native_reference(self):
        raw = native_raw()
        with self.assertRaisesRegex(ValueError, "native reference"):
            step2.detect_flatline_inplace(average(raw.copy()))
        with self.assertRaisesRegex(ValueError, "not entirely zero"):
            step2.detect_flatline_inplace(raw.copy(), reference_channel="E0")
        raw._data[0] = 0
        with self.assertRaisesRegex(ValueError, "exactly one"):
            step2.detect_flatline_inplace(raw.copy())
        # A named reference makes the other all-zero channel an ordinary flat one,
        # which the mask then counts against the pool without the reference.
        flatlines = step2.detect_flatline_inplace(raw, reference_channel="Cz")
        self.assertEqual(flatlines["reference_channel"], "Cz")
        self.assertEqual(flatlines["ch_names"], ["E0"])
        self.assertEqual(flatlines["global_ch_names"], [f"E{i}" for i in range(12)])

    def test_masks_raise_when_no_eligible_eeg_channels_remain(self):
        raw = native_raw()
        flatlines = flatlines_for(raw, {})
        raw.info["bads"] = raw.ch_names[:-1]
        for mask, args in ((step2._mask_segment_by_flatline, (flatlines,)),
                           (step2.mask_segment_by_voltage, ())):
            with self.subTest(mask=mask.__name__), self.assertRaisesRegex(ValueError, "No usable EEG"):
                mask(raw, *args)
        self.assertEqual(len(raw.annotations), 0)

    def test_saved_global_intervals_survive_reference_and_bad_label_changes(self):
        raw = native_raw(n_nonreference=7)
        flatlines = flatlines_for(raw, {name: [[20, 40]] for name in ("E0", "E1", "E2")})
        step2._mask_segment_by_flatline(raw, flatlines)
        raw.info["bads"] = ["E0", "E1", "E2"]
        average(raw)
        self.assertEqual(flatlines["global_segments_exact"], [[20, 40]])
        self.assertEqual(list(_flatline_configurations(raw, flatlines)), [(0, raw.n_times, ())])
        unmasked = native_raw(n_nonreference=12)
        flatlines = flatlines_for(unmasked, {"E0": [[20, 40]], "E1": [[20, 40]]})
        step2._mask_segment_by_flatline(unmasked, flatlines)
        unmasked.info["bads"] = [f"E{i}" for i in range(2, 10)]
        self.assertEqual(flatlines["global_segments_exact"], [])
        self.assertEqual(list(_flatline_configurations(unmasked, flatlines)),
                         [(0, 20, ()), (20, 40, (0, 1)), (40, unmasked.n_times, ())])


class TestLocalFlatlineMetricMasks(unittest.TestCase):
    def test_locally_flat_channel_is_excluded_from_other_channels_metrics(self):
        native = native_raw()
        native._data[0, WINDOW:2 * WINDOW] = 3e-6
        flatlines = flatlines_for(native, {"E0": [[WINDOW, 2 * WINDOW]]})
        referenced = average(native)
        for criterion, detector, metric in DETECTORS:
            with self.subTest(criterion=criterion):
                local = referenced.copy()
                original = local.copy()
                _, masked = detector(local, flatlines=flatlines, return_details=True)
                control = referenced.copy()
                control.info["bads"] = ["E0"]
                _, omitted = detector(control, return_details=True)
                invalid_row = masked["ch_names"].index("E0")
                self.assertTrue(np.isnan(masked[metric][invalid_row, 1]).all())
                self.assertTrue(np.isnan(masked["violates"][invalid_row, 1]))
                for name in omitted["ch_names"]:
                    row = masked["ch_names"].index(name)
                    control_row = omitted["ch_names"].index(name)
                    np.testing.assert_allclose(masked[metric][row, 1], omitted[metric][control_row, 1],
                                               rtol=1e-12, atol=1e-12)
                np.testing.assert_array_equal(local.get_data(), original.get_data())
                self.assertEqual(local.annotations, original.annotations)

    def test_zero_scorable_channel_has_no_metrics_or_automatic_bad_flag(self):
        native = native_raw()
        native._data[0] = 3e-6
        flatlines = flatlines_for(native, {"E0": [[0, native.n_times]]})
        referenced = average(native)
        for criterion, detector, metric in DETECTORS:
            with self.subTest(criterion=criterion):
                report, details = detector(referenced.copy(), flatlines=flatlines, return_details=True)
                row = details["ch_names"].index("E0")
                self.assertNotIn("E0", report[criterion])
                self.assertTrue(np.isnan(details[metric][row]).all())
                self.assertTrue(np.isnan(details["violates"][row]).all())
                self.assertEqual(details["flagged_data_uv"]["E0"].shape, (0, WINDOW))

    def test_occurrence_denominator_is_specific_to_each_channels_scorable_windows(self):
        for criterion, detector, _ in DETECTORS:
            with self.subTest(criterion=criterion):
                native = native_raw()
                if criterion in ("std", "BBnoise"):
                    native._data[0, :WINDOW] *= 12
                elif criterion == "HFnoise":
                    t = np.arange(WINDOW) / SFREQ
                    native._data[0, :WINDOW] += 60e-6 * np.sin(2 * np.pi * 80 * t)
                flatlines = flatlines_for(native, {"E0": [[WINDOW, native.n_times]]})
                referenced = average(native)
                kwargs = {"occurrence_cutoff": 0.5}
                if criterion == "correlation":
                    kwargs["correlation_cutoff"] = 0.9
                report, details = detector(referenced, flatlines=flatlines,
                                           return_details=True, **kwargs)
                self.assertIn("E0", report[criterion])
                row = details["ch_names"].index("E0")
                self.assertEqual(details["violates"][row, 0], 1)
                self.assertTrue(np.isnan(details["violates"][row, 1:]).all())

    def test_std_applies_temporary_voltage_mask_without_mutating_annotations(self):
        raw = average(native_raw())
        raw._data[:5, :WINDOW] += 600e-6
        raw._data[5:10, :WINDOW] -= 600e-6
        _, details = step2.bad_ch_by_std(raw, return_details=True)
        np.testing.assert_array_equal(details["retained"], [False, True, True, True])
        self.assertTrue(np.isnan(details["robust_std_uv"][:, 0]).all())
        self.assertEqual(len(raw.annotations), 0)

    def test_correlation_skips_exact_global_dropout_without_filter_widening(self):
        native = native_raw(n_windows=6)
        flatlines = flatlines_for(native, {name: [[1500, 1520]] for name in ("E0", "E1", "E2", "E3")})
        step2._mask_segment_by_flatline(native, flatlines)
        referenced = average(native)
        original_annotations = referenced.annotations.copy()
        _, details = step2.bad_ch_by_correlation(referenced, flatlines=flatlines, return_details=True)
        expected = ((details["stop_s"] <= 6)
                    | (details["start_s"] >= 6.08))
        np.testing.assert_array_equal(details["retained"], expected)
        self.assertEqual(referenced.annotations, original_annotations)


if __name__ == "__main__":
    unittest.main()
