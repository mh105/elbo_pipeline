import sys
import unittest
from datetime import datetime, timezone
from inspect import signature
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import step2_bad_ch_time_detection as step2


def native_raw(n_times=2000, sfreq=100, first_samp=0):
    data = np.random.default_rng(901).normal(scale=10e-6, size=(6, n_times))
    data[-1] = 0
    return mne.io.RawArray(
        data, mne.create_info([f"E{i}" for i in range(5)] + ["Cz"], sfreq, "eeg"),
        first_samp=first_samp, verbose=False,
    )


def detect(raw):
    with patch.object(step2.mne.filter, "filter_data", side_effect=lambda x, *a, **k: x.copy()):
        return step2.detect_flatline_inplace(raw, "Cz")


def annotation_spans(raw, label):
    return np.column_stack(step2._annotations_starts_stops(raw, [label])).tolist()


class TestFlatlineHandling(unittest.TestCase):
    def test_channel_bridges_and_global_annotation_gaps_are_independent(self):
        raw = native_raw()
        raw._data[:2, 1000:1100] = 5e-6
        raw._data[:2, 1200:1300] = 5e-6
        raw._data[2, 1140:1160] = 7e-6
        original = raw.get_data()
        result = detect(raw)
        self.assertEqual(result["segments_exact"],
                         [[[1000, 1100], [1200, 1300]]] * 2 + [[[1140, 1160]]])
        self.assertEqual(result["segments_bridged"],
                         [[[1000, 1300]]] * 2 + [[[1140, 1160]]])
        self.assertEqual(result["global_segments_exact"], [[1000, 1100], [1200, 1300]])
        self.assertNotIn("segments", result)
        self.assertNotIn("global_segments", result)
        self.assertNotIn("global_segments_bridged", result)
        self.assertEqual(annotation_spans(raw, "BAD_by_flatline"), [[1000, 1100], [1200, 1300]])
        self.assertEqual(annotation_spans(raw, "BAD_by_close_gap"), [[1100, 1200]])
        for row, start, stop in ((0, 1000, 1300), (1, 1000, 1300), (2, 1140, 1160)):
            expected = original[row, start-1] + (original[row, stop] - original[row, start-1]) * np.arange(1, stop-start+1) / (stop-start+1)
            np.testing.assert_allclose(raw._data[row, start:stop], expected, rtol=0, atol=1e-18)
            np.testing.assert_array_equal(raw._data[row, :start], original[row, :start])
            np.testing.assert_array_equal(raw._data[row, stop:], original[row, stop:])
        np.testing.assert_array_equal(raw._data[3:], original[3:])

    def test_global_threshold_precedes_channel_gap_closing(self):
        raw = native_raw()
        raw._data[0, 1000:1100] = 5e-6
        raw._data[0, 1200:1300] = 5e-6
        raw._data[1, 1140:1160] = 7e-6
        result = detect(raw)
        self.assertEqual(result["segments_bridged"][0], [[1000, 1300]])
        self.assertEqual(result["global_segments_exact"], [])
        self.assertEqual(len(raw.annotations), 0)

    def test_exact_two_second_channel_gaps_join_but_one_extra_sample_does_not(self):
        raw = native_raw()
        for start in (500, 710, 921):
            raw._data[0, start:start+10] = 5e-6
        result = detect(raw)
        self.assertEqual(result["segments_exact"], [[[500, 510], [710, 720], [921, 931]]])
        self.assertEqual(result["segments_bridged"], [[[500, 720], [921, 931]]])

    def test_connected_local_global_tail_uses_last_kept_sample(self):
        raw = native_raw()
        raw._data[:5, 1800:] = 0
        raw._data[0, 1700:1800] = 12.5e-6
        original = raw.get_data()
        result = detect(raw)
        self.assertEqual(result["global_segments_exact"], [[1800, 2000]])
        self.assertEqual(result["segments_bridged"][0], [[1700, 2000]])
        np.testing.assert_array_equal(raw._data[0, 1700:], original[0, 1699])
        np.testing.assert_array_equal(raw._data[1:5, 1800:],
                                      np.repeat(original[1:5, [1799]], 200, axis=1))
        np.testing.assert_array_equal(raw._data[-1], 0)

    def test_whole_record_channel_spans_are_left_unchanged(self):
        for kind in ("constant", "ramp", "joined"):
            with self.subTest(kind=kind):
                raw = native_raw(n_times=500)
                if kind == "constant":
                    raw._data[0] = 5e-6
                elif kind == "ramp":
                    raw._data[0] = np.linspace(20e-6, 40e-6, raw.n_times)
                else:
                    raw._data[0, :200] = 5e-6
                    raw._data[0, 300:] = 7e-6
                original = raw.get_data()
                result = detect(raw)
                self.assertEqual(result["segments_bridged"], [[[0, raw.n_times]]])
                self.assertEqual(result["global_segments_exact"], [])
                np.testing.assert_array_equal(raw.get_data(), original)

    def test_detection_does_not_require_a_complete_scoring_window(self):
        parameters = signature(step2.detect_flatline_inplace).parameters
        for name in ("window_length", "highpass_hz", "downstream_highpass_hz"):
            self.assertNotIn(name, parameters)
        raw = native_raw(n_times=400)
        raw._data[:2, 150:160] = 5e-6
        raw._data[:2, 360:370] = 5e-6
        result = detect(raw)
        self.assertEqual(result["global_segments_exact"], [[150, 160], [360, 370]])
        with self.assertRaisesRegex(ValueError, "No complete unmasked windows"):
            step2.bad_ch_by_flatline(raw, reference_channel="Cz", flatlines=result)


class TestBadAnnotationGapPolicy(unittest.TestCase):
    def test_mixed_labels_preserved_and_gap_annotations_rebuilt(self):
        raw = native_raw()
        step2._add_bad_annotations(raw, [[100, 200]], "BAD_by_flatline")
        step2._add_bad_annotations(raw, [[400, 500]], "BAD_manual")
        self.assertEqual(annotation_spans(raw, "BAD_by_close_gap"), [[200, 400]])
        step2._add_bad_annotations(raw, [[250, 300]], "BAD_voltage_mask")
        self.assertEqual(annotation_spans(raw, "BAD_by_close_gap"), [[200, 250], [300, 400]])
        before = raw.annotations.copy()
        step2._add_bad_annotations(raw, [], "BAD_manual")
        self.assertEqual(raw.annotations, before)
        self.assertEqual(annotation_spans(raw, "BAD_by_flatline"), [[100, 200]])
        self.assertEqual(annotation_spans(raw, "BAD_manual"), [[400, 500]])
        self.assertEqual(annotation_spans(raw, "BAD_voltage_mask"), [[250, 300]])

    def test_transitive_gaps_with_crop_time_base_and_measurement_date(self):
        for meas_date in (None, datetime(2026, 10, 12, tzinfo=timezone.utc)):
            with self.subTest(meas_date=meas_date):
                raw = native_raw(first_samp=300)
                raw.set_meas_date(meas_date)
                raw.crop(tmin=0.5)
                step2._add_bad_annotations(raw, [[100, 110], [310, 320], [520, 530], [731, 741]], "bad_manual")
                self.assertEqual(annotation_spans(raw, "BAD_by_close_gap"), [[110, 310], [320, 520]])
                self.assertEqual(raw.annotations.orig_time, meas_date)
                np.testing.assert_allclose(raw.annotations.onset[0], raw.first_time+1)

    def test_zero_duration_and_channel_specific_bad_annotations_do_not_anchor_gaps(self):
        raw = native_raw()
        raw.set_annotations(mne.Annotations([0, 2, 5], [0, 1, 1],
                                            ["BAD_point", "BAD_channel", "BAD_manual"],
                                            ch_names=[(), ("E0",), ()]))
        step2._add_bad_annotations(raw, [[800, 900]], "BAD_by_flatline")
        self.assertEqual(annotation_spans(raw, "BAD_by_close_gap"), [[600, 800]])
        self.assertEqual(raw.annotations.ch_names[1], ("E0",))

    def test_temporary_and_permanent_masks_share_gap_policy_without_flatline_widening(self):
        raw = native_raw()
        step2._add_bad_annotations(raw, [[100, 110]], "BAD_by_flatline")
        raw._data[:2, 310] = 600e-6
        before = raw.annotations.copy()
        spans, _ = step2._temporary_voltage_mask(raw)
        np.testing.assert_array_equal(spans, [[310, 320]])
        rejected = step2._rejected_samples(raw, spans)
        expected = np.zeros(raw.n_times, bool)
        expected[100:320] = True
        np.testing.assert_array_equal(rejected, expected)
        self.assertEqual(raw.annotations, before)
        widened = expected.copy()
        widened[244:386] = True
        np.testing.assert_array_equal(step2._rejected_samples(raw, spans, 66), widened)
        step2.mask_segment_by_voltage(raw)
        np.testing.assert_array_equal(step2._rejected_samples(raw), rejected)
        self.assertEqual(annotation_spans(raw, "BAD_voltage_mask"), [[310, 320]])
        self.assertEqual(annotation_spans(raw, "BAD_by_close_gap"), [[110, 310]])


if __name__ == "__main__":
    unittest.main()
