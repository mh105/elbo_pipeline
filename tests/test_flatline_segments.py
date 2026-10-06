import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import step2_bad_ch_time_detection as step2


def make_raw(n_times=100, n_eeg=8, sfreq=100, first_samp=0, auxiliary=False):
    wave = np.resize([-30e-6, 30e-6], n_times)
    data = np.tile(wave, (n_eeg, 1))
    data[-1] = 0
    names = [f"E{i}" for i in range(n_eeg - 1)] + ["Cz"]
    types = ["eeg"] * n_eeg
    if auxiliary:
        data = np.vstack([wave, data])
        names = ["aux"] + names
        types = ["eog"] + types
    return mne.io.RawArray(
        data, mne.create_info(names, sfreq, types), first_samp=first_samp, verbose=False,
    )


def identity_filter(data, *args, **kwargs):
    return data.copy()


def detect(raw, *args, **kwargs):
    """detect_flatline_inplace keeping every exact detection in segments by
    default: no widening, and no global dropouts (a cutoff of 100% of channels
    is never strictly exceeded), since these recordings have only a few."""
    kwargs.setdefault("downstream_highpass_hz", None)
    kwargs.setdefault("channel_fraction_cutoff", 1.0)
    return step2.detect_flatline_inplace(raw, *args, **kwargs)


def prescribed_segments(raw, by_name):
    """Flatline metadata for raw with prescribed segments; global dropouts are
    decided later by the test, with _mask_segment_by_flatline."""
    with patch.object(step2.mne.filter, "filter_data", side_effect=identity_filter):
        result = detect(raw.copy(), "Cz")
    result["ch_names"] = list(by_name)
    result["segments"] = list(by_name.values())
    result["segments_on_subtypes"] = {
        "segments_on_raw": deepcopy(result["segments"]),
        "segments_on_highpass": [[] for _ in by_name],
        "segments_on_ramp": [[] for _ in by_name],
        "segments_on_curve": [[] for _ in by_name],
    }
    return result


class TestFlatlineDetection(unittest.TestCase):
    def test_ten_samples_qualify_but_nine_do_not(self):
        raw = make_raw(n_times=50)
        raw._data[0, 3:13] = 2e-6
        raw._data[0, 24:33] = 2e-6
        with patch.object(step2.mne.filter, "filter_data", side_effect=identity_filter):
            result = detect(raw, "Cz", channel_fraction_cutoff=0.25)
        self.assertEqual(result["ch_names"], ["E0"])
        self.assertEqual(result["segments"], [[[3, 13]]])
        self.assertNotIn("Cz", result["ch_names"])
        self.assertEqual(result["min_samples"], 10)
        self.assertEqual(result["tolerance_v"], 1e-9)
        self.assertEqual(result["highpass_hz"], 1.0)
        # Detection always ends with the global dropout decision; one flat
        # channel of seven is not a dropout, so nothing is annotated.
        self.assertEqual(result["global_channel_fraction_cutoff"], 0.25)
        self.assertEqual(result["global_ch_names"], [f"E{i}" for i in range(7)])
        self.assertEqual(result["global_segments"], [])
        self.assertEqual(len(raw.annotations), 0)

    def test_custom_tolerance_applies_to_raw_and_highpass_detection(self):
        raw = make_raw(n_times=50, n_eeg=4)
        tolerance_v = 2e-9
        raw._data[0, 5:9] = np.linspace(0, tolerance_v, 4)
        raw._data[1, 15:19] = np.linspace(0, 2.001e-9, 4)
        highpass = raw._data[2].copy()
        highpass[25:29] = np.linspace(0, tolerance_v, 4)

        def filtered(data, *args, **kwargs):
            return highpass.copy() if np.array_equal(data, raw._data[2]) else data.copy()

        with patch.object(step2.mne.filter, "filter_data", side_effect=filtered):
            default = detect(raw, "Cz", min_samples=4)
            result = detect(raw, "Cz", min_samples=4, tolerance_v=tolerance_v)
        self.assertEqual(default["segments"], [])
        self.assertEqual(result["ch_names"], ["E0", "E2"])
        self.assertEqual(result["segments_on_subtypes"]["segments_on_raw"], [[[5, 9]], []])
        self.assertEqual(result["segments_on_subtypes"]["segments_on_highpass"], [[[5, 9]], [[25, 29]]])
        self.assertEqual(result["segments"], [[[5, 9]], [[25, 29]]])
        self.assertEqual(result["tolerance_v"], tolerance_v)

    def test_zero_tolerance_requires_exact_equality(self):
        raw = make_raw(n_times=50, n_eeg=3)
        raw._data[0, 5:15] = 2e-6
        raw._data[1, 25:35] = np.linspace(0, 1e-12, 10)
        with patch.object(step2.mne.filter, "filter_data", side_effect=identity_filter):
            result = detect(raw, "Cz", tolerance_v=0)
        self.assertEqual(result["ch_names"], ["E0"])
        self.assertEqual(result["segments"], [[[5, 15]]])
        self.assertEqual(result["tolerance_v"], 0.0)

    def test_custom_highpass_cutoff_is_used_and_recorded(self):
        raw = make_raw(n_times=50, n_eeg=2)
        highpass = raw._data[0].copy()
        highpass[5:15] = 2e-6
        with patch.object(step2.mne.filter, "filter_data", return_value=highpass) as hp:
            result = detect(raw, "Cz", highpass_hz=2.5)
        self.assertEqual(hp.call_count, 1)
        self.assertEqual(hp.call_args.kwargs, {"l_freq": 2.5, "h_freq": None, "verbose": False})
        self.assertEqual(result["segments_on_subtypes"]["segments_on_raw"], [[]])
        self.assertEqual(result["segments_on_subtypes"]["segments_on_highpass"], [[[5, 15]]])
        self.assertEqual(result["highpass_hz"], 2.5)

    def test_tolerance_and_highpass_cutoff_require_valid_ranges(self):
        raw = make_raw()
        for name, values in (("tolerance_v", (-1e-9, np.nan, np.inf)),
                             ("highpass_hz", (0, -1, 50, 51, np.nan, np.inf))):
            for value in values:
                with self.subTest(parameter=name, value=value), self.assertRaisesRegex(ValueError, name):
                    detect(raw, "Cz", **{name: value})

    def test_custom_min_samples_applies_to_raw_and_highpass_detection(self):
        for min_samples in (5, 6, 15, np.int64(7)):
            with self.subTest(min_samples=min_samples):
                raw = make_raw(n_times=70, n_eeg=4)
                raw._data[0, 3:3 + min_samples] = 2e-6
                raw._data[1, 35:35 + min_samples - 1] = 4e-6
                highpass = raw._data[1].copy()
                highpass[45:45 + min_samples] = 3e-6

                def filtered(data, *args, **kwargs):
                    return highpass.copy() if np.array_equal(data, raw._data[1]) else data.copy()

                with patch.object(step2.mne.filter, "filter_data", side_effect=filtered):
                    result = detect(raw, "Cz", min_samples=min_samples)
                self.assertEqual(result["ch_names"], ["E0", "E1"])
                self.assertEqual(result["segments_on_subtypes"]["segments_on_raw"],
                                 [[[3, 3 + min_samples]], []])
                expected = [[[3, 3 + min_samples]], [[45, 45 + min_samples]]]
                self.assertEqual(result["segments_on_subtypes"]["segments_on_highpass"], expected)
                self.assertEqual(result["segments"], expected)
                self.assertEqual(result["min_samples"], min_samples)
                self.assertIs(type(result["min_samples"]), int)

    def test_custom_min_samples_handles_short_recordings(self):
        for min_samples in (2, 3, 4):
            with self.subTest(min_samples=min_samples):
                # One flat channel of seven, so it is not a global dropout.
                raw = make_raw(n_times=3, n_eeg=8)
                raw._data[0] = 2e-6
                with patch.object(step2.mne.filter, "filter_data", side_effect=identity_filter) as hp:
                    result = detect(raw, "Cz", min_samples=min_samples)
                qualifies = min_samples <= raw.n_times
                self.assertEqual(result["segments"], [[[0, 3]]] if qualifies else [])
                self.assertEqual(hp.call_count, 7 * int(qualifies))

    def test_large_min_samples_preserves_one_contiguous_flat_segment(self):
        raw = make_raw(n_times=65536, n_eeg=8)  # One flat channel of seven.
        raw._data[0] = 2e-6
        with patch.object(step2.mne.filter, "filter_data", side_effect=identity_filter):
            result = detect(raw, "Cz", min_samples=32768)
        self.assertEqual(result["ch_names"], ["E0"])
        self.assertEqual(result["segments"], [[[0, raw.n_times]]])

    def test_min_samples_requires_an_integer_of_at_least_two(self):
        raw = make_raw()
        # One sample always has zero range, so min_samples=1 would mark everything flat.
        for min_samples in (1, 0, -1, 2.5, 10.0, np.nan, np.inf, "10", None, True,
                            np.bool_(True)):
            with self.subTest(min_samples=min_samples), self.assertRaisesRegex(ValueError, "min_samples"):
                detect(raw, "Cz", min_samples=min_samples)

    def test_full_range_includes_tolerance_and_does_not_chain_adjacent_differences(self):
        raw = make_raw(n_times=50)
        raw._data[0, 5:15] = np.linspace(0, 1e-9, 10)
        raw._data[1, 5:15] = np.linspace(0, 1.001e-9, 10)
        raw._data[2, 5:25] = np.arange(20) * 0.4e-9
        with patch.object(step2.mne.filter, "filter_data", side_effect=identity_filter):
            result = detect(raw, "Cz")
        subtypes = result["segments_on_subtypes"]
        self.assertEqual(result["ch_names"], ["E0", "E2"])
        # The raw rule does not chain E2's small adjacent differences into one
        # block; E2's 20-sample exact ramp is caught by the ramp and curve rules.
        self.assertEqual(subtypes["segments_on_raw"], [[[5, 15]], []])
        self.assertEqual(subtypes["segments_on_ramp"], [[], [[5, 25]]])
        self.assertEqual(subtypes["segments_on_curve"], [[], [[5, 25]]])
        self.assertEqual(result["segments"], [[[5, 15]], [[5, 25]]])

    def test_ramp_and_curve_rules_use_blocks_of_twice_min_samples(self):
        raw = make_raw(n_times=600, n_eeg=4, sfreq=1000)
        raw._data[0, 50:70] = 5e-6 + np.arange(20) * 0.1e-6  # 20-sample exact ramp.
        raw._data[1, 50:69] = 5e-6 + np.arange(19) * 0.1e-6  # One sample too short.
        # A 1-mV decay with a 1.5-s time constant: its second differences stay
        # below 1 nV, while its first differences change by about 8 nV per block.
        t = np.arange(300) / 1000
        raw._data[2, 100:400] = 1e-3 * np.exp(-t / 1.5)
        with patch.object(step2.mne.filter, "filter_data", side_effect=identity_filter):
            result = detect(raw, "Cz")
        subtypes = result["segments_on_subtypes"]
        self.assertEqual(result["ch_names"], ["E0", "E2"])
        self.assertEqual(subtypes["segments_on_raw"], [[], []])
        self.assertEqual(subtypes["segments_on_ramp"], [[[50, 70]], []])
        self.assertEqual(subtypes["segments_on_curve"], [[[50, 70]], [[100, 400]]])
        self.assertEqual(result["segments"], [[[50, 70]], [[100, 400]]])

    def test_raw_and_highpass_segments_union_with_matching_audit_rows(self):
        raw = make_raw(n_times=60)
        raw._data[0, 5:15] = 2e-6
        highpass = raw._data[0].copy()
        highpass[13:24] = 2e-6
        def filtered(data, *args, **kwargs):
            return highpass.copy() if np.array_equal(data, raw._data[0]) else data.copy()
        with patch.object(step2.mne.filter, "filter_data", side_effect=filtered):
            result = detect(raw, "Cz")
        self.assertEqual(result["segments_on_subtypes"], {
            "segments_on_raw": [[[5, 15]]],
            "segments_on_highpass": [[[5, 24]]],
            "segments_on_ramp": [[]],
            "segments_on_curve": [[]],
        })
        for key in ("raw_segments", "highpass_segments"):
            self.assertNotIn(key, result)
        self.assertEqual(result["segments"], [[[5, 24]]])

    def test_highpass_detects_drift_without_changing_raw(self):
        raw = make_raw(n_times=2000)
        raw._data[0] = 10e-6 + np.arange(raw.n_times) / raw.info["sfreq"] * 1e-6
        raw.info["bads"] = ["E1"]
        raw.set_annotations(mne.Annotations([2], [1], ["BAD_test"]))
        original = raw.copy()
        result = detect(raw, "Cz")
        row = result["ch_names"].index("E0")
        self.assertEqual(result["segments_on_subtypes"]["segments_on_raw"][row], [])
        self.assertTrue(any(start <= 500 and stop >= 1500
                            for start, stop in result["segments_on_subtypes"]["segments_on_highpass"][row]))
        np.testing.assert_array_equal(raw.get_data(), original.get_data())
        self.assertEqual(raw.info["bads"], original.info["bads"])
        self.assertEqual(raw.annotations, original.annotations)

    def test_finite_existing_bads_are_detected_but_nonfinite_bads_are_skipped(self):
        raw = make_raw()
        raw._data[0, 20:40] = 3e-6
        raw._data[1, 10] = np.nan
        raw.info["bads"] = ["E0", "E1"]
        with patch.object(step2.mne.filter, "filter_data", side_effect=identity_filter) as hp:
            result = detect(raw, "Cz")
        self.assertEqual(result["ch_names"], ["E0"])
        self.assertEqual(len(result["eeg_ch_names"]), 8)
        self.assertEqual(hp.call_count, 6)
        raw.info["bads"] = ["E0"]
        with patch.object(step2.mne.filter, "filter_data", side_effect=identity_filter):
            with self.assertRaisesRegex(ValueError, "bad_ch_by_nan"):
                detect(raw, "Cz")

    def test_ten_samples_do_not_depend_on_sampling_rate_or_reference_bad_label(self):
        for sfreq in (100, 1000):
            with self.subTest(sfreq=sfreq):
                raw = make_raw(n_times=10, sfreq=sfreq)
                raw._data[0] = 0
                raw.info["bads"] = ["Cz"]
                with patch.object(step2.mne.filter, "filter_data", side_effect=identity_filter):
                    result = detect(raw, "Cz")
                self.assertEqual(result["ch_names"], ["E0"])
                self.assertEqual(result["segments"], [[[0, 10]]])
        raw = make_raw(n_times=9)
        raw._data[0] = 0
        with patch.object(step2.mne.filter, "filter_data") as hp:
            result = detect(raw, "Cz")
        self.assertEqual(result["ch_names"], [])
        self.assertEqual(result["segments_on_subtypes"], {
            "segments_on_raw": [],
            "segments_on_highpass": [],
            "segments_on_ramp": [],
            "segments_on_curve": [],
        })
        hp.assert_not_called()


class TestFlatlineBridgingAndWidening(unittest.TestCase):
    def setUp(self):
        # 12 eligible channels at 100 Hz: 4 flat at once (> 25%) are a global dropout.
        self.raw = make_raw(n_times=3000, n_eeg=13, sfreq=100)
        data = self.raw._data
        data[0, 1000:1100] = 50e-6  # Local span inside the recording.
        data[1, :100] = 20e-6  # Local span at the start...
        data[2, 2900:] = -20e-6  # ...and at the end.
        data[3:7, 2000:2200] = 5e-6  # Global dropout on E3-E6.
        data[3, 2200:2300] = 5e-6  # E3 stays flat 1 s longer: a local tail.
        self.before = self.raw.copy()

    def detect(self, **kwargs):
        with patch.object(step2.mne.filter, "filter_data", side_effect=identity_filter):
            return step2.detect_flatline_inplace(self.raw, "Cz", **kwargs)

    def test_local_spans_are_bridged_but_global_dropouts_are_not(self):
        result = self.detect()
        x, original = self.raw._data, self.before._data
        line = original[0, 999] + (original[0, 1100] - original[0, 999]) * np.arange(1, 101) / 101
        np.testing.assert_allclose(x[0, 1000:1100], line, rtol=0, atol=1e-18)
        np.testing.assert_array_equal(x[1, :100], original[1, 100])
        np.testing.assert_array_equal(x[2, 2900:], original[2, 2899])
        np.testing.assert_array_equal(x[3:7, 2000:2200], original[3:7, 2000:2200])
        tail = original[3, 2199] + (original[3, 2300] - original[3, 2199]) * np.arange(1, 101) / 101
        np.testing.assert_allclose(x[3, 2200:2300], tail, rtol=0, atol=1e-18)
        changed = (x != original).any(axis=0)
        np.testing.assert_array_equal(np.flatnonzero(changed), np.r_[0:100, 1000:1100, 2200:2300, 2900:3000])
        self.assertEqual(result["global_segments"], [[2000, 2200]])
        self.assertEqual(result["segments_on_subtypes"]["segments_on_raw"][3], [[2000, 2300]])

    def test_local_intervals_and_annotation_are_widened_by_the_highpass_reach(self):
        result = self.detect()
        reach = 550  # MNE's default 0.3 Hz FIR is 11 s long: 5.5 s on each side.
        self.assertEqual((result["downstream_highpass_hz"], result["widen_samples"]), (0.3, reach))
        segments = dict(zip(result["ch_names"], result["segments"]))
        self.assertEqual(segments, {
            "E0": [[1000 - reach, 1100 + reach]], "E1": [[0, 100 + reach]],
            "E2": [[2900 - reach, 3000]], "E3": [[2200 - reach, 2300 + reach]],
            "E4": [], "E5": [], "E6": [],  # Flat only inside the dropout.
        })
        np.testing.assert_array_equal(self.raw.annotations.description, ["BAD_by_flatline"])
        np.testing.assert_allclose(self.raw.annotations.onset, [(2000 - reach) / 100])
        np.testing.assert_allclose(self.raw.annotations.duration, [(200 + 2 * reach) / 100])

    def test_widening_can_be_skipped_and_reach_is_5_5_s_at_any_rate(self):
        result = self.detect(downstream_highpass_hz=None)
        self.assertEqual((result["downstream_highpass_hz"], result["widen_samples"]), (None, 0))
        self.assertEqual(result["segments"][0], [[1000, 1100]])
        np.testing.assert_allclose(self.raw.annotations.onset, [20])
        np.testing.assert_allclose(self.raw.annotations.duration, [2])
        for sfreq in (250, 500, 1000):
            with self.subTest(sfreq=sfreq):
                raw = make_raw(n_times=50, sfreq=sfreq)
                with patch.object(step2.mne.filter, "filter_data", side_effect=identity_filter):
                    result = step2.detect_flatline_inplace(raw, "Cz")
                self.assertEqual(result["widen_samples"], int(5.5 * sfreq))

    def test_invalid_widening_and_unloaded_data_raise(self):
        for value in (0, -1, 50, np.nan, np.inf):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "downstream_highpass_hz"
            ):
                self.detect(downstream_highpass_hz=value)
        unloaded = make_raw(n_times=50)
        unloaded.preload = False
        with self.assertRaisesRegex(ValueError, "preloaded"):
            step2.detect_flatline_inplace(unloaded, "Cz")


class TestFlatlineApplication(unittest.TestCase):
    def test_detection_masks_global_dropouts_in_place(self):
        raw = make_raw(n_times=50, n_eeg=9)
        raw.info["bads"] = ["E7"]
        raw._data[:2, 10:20] = 2e-6  # E0 and E1: 2 of 7 eligible channels.
        with patch.object(step2.mne.filter, "filter_data", side_effect=identity_filter):
            result = detect(raw, "Cz", channel_fraction_cutoff=0.2)
        self.assertEqual(result["global_channel_fraction_cutoff"], 0.2)
        self.assertEqual(result["global_ch_names"], [f"E{i}" for i in range(7)])
        self.assertEqual(result["global_segments"], [[10, 20]])
        np.testing.assert_array_equal(raw.annotations.description, ["BAD_by_flatline"])
        np.testing.assert_allclose(raw.annotations.onset, [0.1])
        np.testing.assert_allclose(raw.annotations.duration, [0.1])
        for cutoff in (-0.1, 1.1, np.nan):
            with self.subTest(cutoff=cutoff), self.assertRaisesRegex(
                ValueError, "channel_fraction_cutoff"
            ):
                detect(raw.copy(), "Cz", channel_fraction_cutoff=cutoff)

    def test_window_overlap_half_open_boundaries_and_partial_tail(self):
        mask = step2._segments_to_windows([[19, 30], [40, 49]], 10, 4)
        np.testing.assert_array_equal(mask, [False, True, True, False])
        np.testing.assert_array_equal(step2._segments_to_windows([[20, 30]], 10, 4),
                                      [False, False, True, False])

    def test_masks_follow_picks_and_metadata_allow_filtering_and_bad_label_changes(self):
        raw = make_raw(n_times=50, first_samp=500, auxiliary=True)
        result = prescribed_segments(raw, {"E0": [[19, 30]], "E1": [[35, 49]]})
        raw.info["bads"] = ["E0", "Cz"]
        raw._data += 1e-6  # Referencing/filtering preserves alignment, not amplitudes.
        step2._validate_flatlines(raw, result)
        picks = [raw.ch_names.index(name) for name in ("E1", "Cz", "E0")]
        mask = step2._flatline_windows(raw, result, picks, 10, 4)
        np.testing.assert_array_equal(mask,
                                      [[False, False, False, True], [False] * 4,
                                       [False, True, True, False]])
        np.testing.assert_array_equal(step2._flatline_windows(raw, None, picks, 10, 4),
                                      np.zeros((3, 4), dtype=bool))

    def test_metadata_rejects_crop_resampling_channel_reordering_and_malformed_segments(self):
        raw = make_raw(n_times=100)
        result = prescribed_segments(raw, {"E0": [[10, 20]]})
        changed = [raw.copy().crop(tmin=0.1), raw.copy().resample(50, verbose=False),
                   raw.copy().reorder_channels(list(reversed(raw.ch_names)))]
        for other in changed:
            with self.subTest(other=other), self.assertRaisesRegex(ValueError, "alignment"):
                step2._validate_flatlines(other, result)
        for intervals in ([[20, 10]], [[10.5, 20]], [[10, 101]], [[10, 30], [20, 40]]):
            malformed = deepcopy(result)
            malformed["segments"] = [intervals]
            with self.subTest(intervals=intervals), self.assertRaisesRegex(ValueError, "intervals"):
                step2._validate_flatlines(raw, malformed)

    def test_global_threshold_uses_only_eligible_channels_and_strict_boundary(self):
        raw = make_raw(n_times=50, n_eeg=9)
        result = prescribed_segments(raw, {
            "E0": [[4, 20]], "E1": [[10, 30]], "E2": [[12, 18]],
        })
        names = [raw.ch_names[i] for i in step2._mask_eeg_picks(raw, "Cz")]
        self.assertEqual(step2._global_flatline_segments(result, names), [[12, 18]])
        result["segments"][2] = []
        self.assertEqual(step2._global_flatline_segments(result, names), [])  # 2/8 == 25%.
        raw.info["bads"] = ["E3", "E4"]
        names = [raw.ch_names[i] for i in step2._mask_eeg_picks(raw, "Cz")]
        self.assertEqual(step2._global_flatline_segments(result, names), [[10, 20]])
        raw.info["bads"] = ["E0", "E3", "E4"]
        names = [raw.ch_names[i] for i in step2._mask_eeg_picks(raw, "Cz")]
        self.assertEqual(step2._global_flatline_segments(result, names), [])
        for cutoff in (-0.1, 1.1, np.nan):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                step2._global_flatline_segments(result, names, cutoff)

    def test_configurations_cover_recording_use_full_raw_indices_and_omit_global_spans(self):
        raw = make_raw(n_times=50, auxiliary=True)
        result = prescribed_segments(raw, {
            "E0": [[4, 20]], "E1": [[10, 30]], "E2": [[12, 18]],
        })
        step2._mask_segment_by_flatline(raw, result)  # Global dropout [10, 20).
        expected = [(0, 4, ()), (4, 10, (1,)), (10, 20, ()),
                    (20, 30, (2,)), (30, 50, ())]
        self.assertEqual(list(step2._flatline_configurations(raw, result)), expected)
        self.assertEqual(list(step2._flatline_configurations(raw, None)), [(0, 50, ())])
        del result["global_segments"]
        with self.assertRaisesRegex(ValueError, "global_segments"):
            list(step2._flatline_configurations(raw, result))


if __name__ == "__main__":
    unittest.main()
