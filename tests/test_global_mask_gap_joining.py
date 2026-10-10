import sys
import unittest
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import step2_bad_ch_time_detection as step2


def native_raw(n_times=1600, sfreq=100, first_samp=0):
    data = np.ones((9, n_times)) * 1e-6
    data[-1] = 0
    return mne.io.RawArray(
        data, mne.create_info([f"E{i}" for i in range(8)] + ["Cz"], sfreq, "eeg"),
        first_samp=first_samp, verbose=False,
    )


def flatlines_for(spans):
    return {
        "reference_channel": "Cz", "ch_names": ["E0", "E1", "E2"],
        "segments": [deepcopy(spans) for _ in range(3)],
    }


def voltage_annotation_spans(raw):
    starts, stops = step2._annotations_starts_stops(raw, ["BAD_voltage_mask"])
    return np.column_stack([starts, stops])


class TestVoltageGapJoining(unittest.TestCase):
    def test_default_mask_joins_two_second_gaps_transitively(self):
        # At 10 Hz, the default 0.1-s mask windows are individual samples.
        raw = native_raw(n_times=100, sfreq=10)
        raw._data[:3, [10, 31, 52, 74]] = 501e-6
        spans, reference = step2._temporary_voltage_mask(raw)
        np.testing.assert_array_equal(spans, [[10, 53], [74, 75]])
        self.assertEqual(reference, "Cz")
        self.assertIsNone(step2.mask_segment_by_voltage(raw))
        np.testing.assert_array_equal(voltage_annotation_spans(raw), spans)

    def test_custom_public_settings_match_cached_temporary_mask(self):
        raw = native_raw()
        raw._data[0, [100, 325]] = 126e-6
        shared = step2._mask_inputs(raw, voltage_threshold_uv=125, window_length=0.25)
        spans, _ = step2._temporary_voltage_mask(
            raw, reference_channel="Cz", shared=shared, channel_fraction_cutoff=0.1,
        )
        np.testing.assert_array_equal(spans, [[100, 350]])
        step2.mask_segment_by_voltage(
            raw, voltage_threshold_uv=125, channel_fraction_cutoff=0.1,
            window_length=0.25, reference_channel="Cz",
        )
        np.testing.assert_array_equal(voltage_annotation_spans(raw), spans)

    def test_cache_reuses_flags_but_recalculates_the_eligible_denominator(self):
        raw = native_raw()
        raw._data[:2, 101] = 501e-6  # Two of eight equals the cutoff.
        shared = step2._mask_inputs(raw)
        with patch.object(raw, "get_data", side_effect=AssertionError("Data rescanned")):
            spans, _ = step2._temporary_voltage_mask(raw, shared=shared)
            self.assertEqual(spans.shape, (0, 2))
            raw.info["bads"] = ["E7"]
            spans, _ = step2._temporary_voltage_mask(raw, shared=shared)
            np.testing.assert_array_equal(spans, [[100, 110]])  # Two of seven exceeds it.
            raw.info["bads"] = ["E7", "E0"]  # Exclude one exceeding channel too.
            spans, _ = step2._temporary_voltage_mask(raw, shared=shared)
            self.assertEqual(spans.shape, (0, 2))  # One of six is below the cutoff.

    def test_noninteger_sampling_rate_does_not_round_up_the_gap_limit(self):
        raw = native_raw(n_times=100, sfreq=10.4)
        raw._data[:3, [10, 31, 53]] = 501e-6
        spans, _ = step2._temporary_voltage_mask(raw)
        # Twenty samples are <2 s; twenty-one samples are >2 s.
        np.testing.assert_array_equal(spans, [[10, 32], [53, 54]])

    def test_excluded_nan_channel_does_not_reject_the_mask(self):
        raw = native_raw()
        raw._data[7] = np.nan
        raw.info["bads"] = ["E7"]
        raw._data[:2, 101] = 501e-6
        shared = step2._mask_inputs(raw)
        spans, _ = step2._temporary_voltage_mask(raw, shared=shared)
        np.testing.assert_array_equal(spans, [[100, 110]])
        step2.mask_segment_by_voltage(raw)
        np.testing.assert_array_equal(voltage_annotation_spans(raw), spans)

    def test_temporary_mask_keeps_data_bads_and_annotations_unchanged(self):
        raw = native_raw()
        raw._data[:3, [101, 311]] = 501e-6
        raw.set_annotations(mne.Annotations([0.2, 9], [0, 0.1], ["event", "BAD_manual"]))
        before = raw.copy()
        spans, _ = step2._temporary_voltage_mask(raw)
        np.testing.assert_array_equal(spans, [[100, 320]])
        np.testing.assert_array_equal(raw.get_data(), before.get_data())
        self.assertEqual(raw.info["bads"], before.info["bads"])
        self.assertEqual(raw.annotations, before.annotations)

    def test_public_and_temporary_reject_conflicting_reference_requirements(self):
        raw = native_raw()
        for function in (step2.mask_segment_by_voltage, step2._temporary_voltage_mask):
            with self.subTest(function=function.__name__), self.assertRaisesRegex(
                ValueError, "require_average"
            ):
                function(raw, reference_channel="Cz", require_average=True)

    def test_cache_validates_voltage_threshold_and_window_length(self):
        raw = native_raw()
        for settings in (
            {"voltage_threshold_uv": 0}, {"voltage_threshold_uv": np.inf},
            {"window_length": 0.001}, {"window_length": np.nan},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                step2._mask_inputs(raw, **settings)

    def test_cropped_annotation_time_base_matches_temporary_spans(self):
        for meas_date in (None, datetime(2026, 10, 9, tzinfo=timezone.utc)):
            with self.subTest(meas_date=meas_date):
                raw = native_raw(n_times=1800, first_samp=250)
                raw.set_meas_date(meas_date)
                raw._data[:3, [601, 811]] = 501e-6
                raw.crop(tmin=0.5)
                spans, _ = step2._temporary_voltage_mask(raw)
                np.testing.assert_array_equal(spans, [[550, 770]])
                step2.mask_segment_by_voltage(raw)
                np.testing.assert_array_equal(voltage_annotation_spans(raw), spans)
                np.testing.assert_allclose(raw.annotations.onset, [raw.first_time + 5.5])
                np.testing.assert_allclose(raw.annotations.duration, [2.2])
                self.assertEqual(raw.annotations.orig_time, meas_date)


class TestFlatlineGapJoining(unittest.TestCase):
    def test_exact_two_second_gaps_join_and_one_extra_sample_does_not(self):
        raw = native_raw()
        exact = [[100, 110], [310, 320], [521, 531]]
        flatlines = flatlines_for(exact)
        step2._mask_segment_by_flatline(raw, flatlines)
        starts, stops = step2._annotations_starts_stops(raw, ["BAD_by_flatline"])
        np.testing.assert_array_equal(np.column_stack([starts, stops]),
                                      [[100, 320], [521, 531]])
        self.assertEqual(flatlines["global_segments"], exact)
        self.assertEqual(flatlines["segments"], [exact] * 3)

    def test_annotation_gap_is_joined_after_widening(self):
        raw = native_raw()
        exact = [[100, 110], [350, 360]]  # 2.4-s gap before widening.
        flatlines = flatlines_for(exact)
        step2._mask_segment_by_flatline(raw, flatlines, widen_samples=20)
        starts, stops = step2._annotations_starts_stops(raw, ["BAD_by_flatline"])
        np.testing.assert_array_equal(np.column_stack([starts, stops]), [[80, 380]])
        self.assertEqual(flatlines["global_segments"], exact)

    def test_joining_checks_scorable_windows_before_mutating_inputs(self):
        raw = native_raw(n_times=600)
        # The middle 2-s window is clear before joining these two dropouts.
        flatlines = flatlines_for([[0, 200], [400, 600]])
        before = deepcopy(flatlines)
        with self.assertRaisesRegex(ValueError, "no complete.*2-s window|entire recording"):
            step2._mask_segment_by_flatline(raw, flatlines)
        self.assertEqual(flatlines, before)
        self.assertEqual(len(raw.annotations), 0)

    def test_joined_annotations_preserve_local_flatline_bridging_in_the_gap(self):
        raw = native_raw()
        raw._data[:8] = np.random.default_rng(21).normal(scale=10e-6, size=(8, 1600))
        raw._data[:3, 100:110] = 1e-6
        raw._data[:3, 310:320] = 1e-6
        raw._data[3, 200:220] = 1e-6
        before = raw.get_data().copy()
        flatlines = step2.detect_flatline_inplace(raw, downstream_highpass_hz=None)
        self.assertEqual(flatlines["global_segments"], [[100, 110], [310, 320]])
        starts, stops = step2._annotations_starts_stops(raw, ["BAD_by_flatline"])
        np.testing.assert_array_equal(np.column_stack([starts, stops]), [[100, 320]])
        expected = before[3, 199] + (before[3, 220] - before[3, 199]) * np.arange(1, 21) / 21
        np.testing.assert_allclose(raw._data[3, 200:220], expected, rtol=0, atol=1e-18)
        self.assertIn((200, 220, (3,)), list(step2._flatline_configurations(raw, flatlines)))
        np.testing.assert_array_equal(raw._data[:3, 100:110], before[:3, 100:110])


if __name__ == "__main__":
    unittest.main()
