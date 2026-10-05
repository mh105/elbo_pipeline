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

import flatline


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


def prescribed_segments(raw, by_name):
    with patch.object(flatline.mne.filter, "filter_data", side_effect=identity_filter):
        result = flatline._detect_flatline_data(raw, "Cz")
    result["ch_names"] = list(by_name)
    result["segments"] = list(by_name.values())
    result["raw_segments"] = deepcopy(result["segments"])
    result["highpass_segments"] = [[] for _ in by_name]
    return result


class TestFlatlineDetection(unittest.TestCase):
    def test_ten_samples_qualify_but_nine_do_not(self):
        raw = make_raw(n_times=50)
        raw._data[0, 3:13] = 2e-6
        raw._data[0, 24:33] = 2e-6
        with patch.object(flatline.mne.filter, "filter_data", side_effect=identity_filter):
            result = flatline._detect_flatline_data(raw, "Cz")
        self.assertEqual(result["ch_names"], ["E0"])
        self.assertEqual(result["segments"], [[[3, 13]]])
        self.assertNotIn("Cz", result["ch_names"])
        self.assertEqual(result["min_samples"], 10)
        self.assertEqual(result["tolerance_v"], 1e-9)

    def test_full_range_includes_tolerance_and_does_not_chain_adjacent_differences(self):
        raw = make_raw(n_times=50)
        raw._data[0, 5:15] = np.linspace(0, 1e-9, 10)
        raw._data[1, 5:15] = np.linspace(0, 1.001e-9, 10)
        raw._data[2, 5:25] = np.arange(20) * 0.4e-9
        with patch.object(flatline.mne.filter, "filter_data", side_effect=identity_filter):
            result = flatline._detect_flatline_data(raw, "Cz")
        self.assertEqual(result["ch_names"], ["E0"])
        self.assertEqual(result["segments"], [[[5, 15]]])

    def test_raw_and_highpass_segments_union_with_matching_audit_rows(self):
        raw = make_raw(n_times=60)
        raw._data[0, 5:15] = 2e-6
        highpass = raw._data[0].copy()
        highpass[13:24] = 2e-6
        def filtered(data, *args, **kwargs):
            return highpass.copy() if np.array_equal(data, raw._data[0]) else data.copy()
        with patch.object(flatline.mne.filter, "filter_data", side_effect=filtered):
            result = flatline._detect_flatline_data(raw, "Cz")
        self.assertEqual(result["raw_segments"], [[[5, 15]]])
        self.assertEqual(result["highpass_segments"], [[[5, 24]]])
        self.assertEqual(result["segments"], [[[5, 24]]])

    def test_highpass_detects_drift_without_changing_raw(self):
        raw = make_raw(n_times=2000)
        raw._data[0] = 10e-6 + np.arange(raw.n_times) / raw.info["sfreq"] * 1e-6
        raw.info["bads"] = ["E1"]
        raw.set_annotations(mne.Annotations([2], [1], ["BAD_test"]))
        original = raw.copy()
        result = flatline._detect_flatline_data(raw, "Cz")
        row = result["ch_names"].index("E0")
        self.assertEqual(result["raw_segments"][row], [])
        self.assertTrue(any(start <= 500 and stop >= 1500
                            for start, stop in result["highpass_segments"][row]))
        np.testing.assert_array_equal(raw.get_data(), original.get_data())
        self.assertEqual(raw.info["bads"], original.info["bads"])
        self.assertEqual(raw.annotations, original.annotations)

    def test_finite_existing_bads_are_detected_but_nonfinite_bads_are_skipped(self):
        raw = make_raw()
        raw._data[0, 20:40] = 3e-6
        raw._data[1, 10] = np.nan
        raw.info["bads"] = ["E0", "E1"]
        with patch.object(flatline.mne.filter, "filter_data", side_effect=identity_filter) as hp:
            result = flatline._detect_flatline_data(raw, "Cz")
        self.assertEqual(result["ch_names"], ["E0"])
        self.assertEqual(len(result["eeg_ch_names"]), 8)
        self.assertEqual(hp.call_count, 6)
        raw.info["bads"] = ["E0"]
        with patch.object(flatline.mne.filter, "filter_data", side_effect=identity_filter):
            with self.assertRaisesRegex(ValueError, "bad_ch_by_nan"):
                flatline._detect_flatline_data(raw, "Cz")

    def test_ten_samples_do_not_depend_on_sampling_rate_or_reference_bad_label(self):
        for sfreq in (100, 1000):
            with self.subTest(sfreq=sfreq):
                raw = make_raw(n_times=10, sfreq=sfreq)
                raw._data[0] = 0
                raw.info["bads"] = ["Cz"]
                with patch.object(flatline.mne.filter, "filter_data", side_effect=identity_filter):
                    result = flatline._detect_flatline_data(raw, "Cz")
                self.assertEqual(result["ch_names"], ["E0"])
                self.assertEqual(result["segments"], [[[0, 10]]])
        raw = make_raw(n_times=9)
        raw._data[0] = 0
        with patch.object(flatline.mne.filter, "filter_data") as hp:
            result = flatline._detect_flatline_data(raw, "Cz")
        self.assertEqual(result["ch_names"], [])
        hp.assert_not_called()


class TestFlatlineApplication(unittest.TestCase):
    def test_window_overlap_half_open_boundaries_and_partial_tail(self):
        mask = flatline._segments_to_windows([[19, 30], [40, 49]], 10, 4)
        np.testing.assert_array_equal(mask, [False, True, True, False])
        np.testing.assert_array_equal(flatline._segments_to_windows([[20, 30]], 10, 4),
                                      [False, False, True, False])

    def test_masks_follow_picks_and_metadata_allow_filtering_and_bad_label_changes(self):
        raw = make_raw(n_times=50, first_samp=500, auxiliary=True)
        result = prescribed_segments(raw, {"E0": [[19, 30]], "E1": [[35, 49]]})
        raw.info["bads"] = ["E0", "Cz"]
        raw._data += 1e-6  # Referencing/filtering preserves alignment, not amplitudes.
        flatline._validate_flatlines(raw, result)
        picks = [raw.ch_names.index(name) for name in ("E1", "Cz", "E0")]
        mask = flatline._flatline_windows(raw, result, picks, 10, 4)
        np.testing.assert_array_equal(mask,
                                      [[False, False, False, True], [False] * 4,
                                       [False, True, True, False]])
        np.testing.assert_array_equal(flatline._flatline_windows(raw, None, picks, 10, 4),
                                      np.zeros((3, 4), dtype=bool))

    def test_metadata_rejects_crop_resampling_channel_reordering_and_malformed_segments(self):
        raw = make_raw(n_times=100)
        result = prescribed_segments(raw, {"E0": [[10, 20]]})
        changed = [raw.copy().crop(tmin=0.1), raw.copy().resample(50, verbose=False),
                   raw.copy().reorder_channels(list(reversed(raw.ch_names)))]
        for other in changed:
            with self.subTest(other=other), self.assertRaisesRegex(ValueError, "alignment"):
                flatline._validate_flatlines(other, result)
        for intervals in ([[20, 10]], [[10.5, 20]], [[10, 101]], [[10, 30], [20, 40]]):
            malformed = deepcopy(result)
            malformed["segments"] = [intervals]
            with self.subTest(intervals=intervals), self.assertRaisesRegex(ValueError, "intervals"):
                flatline._validate_flatlines(raw, malformed)

    def test_global_threshold_uses_only_eligible_channels_and_strict_boundary(self):
        raw = make_raw(n_times=50, n_eeg=9)
        result = prescribed_segments(raw, {
            "E0": [[4, 20]], "E1": [[10, 30]], "E2": [[12, 18]],
        })
        names = [raw.ch_names[i] for i in flatline._mask_eeg_picks(raw, "Cz")]
        self.assertEqual(flatline._global_flatline_segments(result, names), [[12, 18]])
        result["segments"][2] = []
        self.assertEqual(flatline._global_flatline_segments(result, names), [])  # 2/8 == 25%.
        raw.info["bads"] = ["E3", "E4"]
        names = [raw.ch_names[i] for i in flatline._mask_eeg_picks(raw, "Cz")]
        self.assertEqual(flatline._global_flatline_segments(result, names), [[10, 20]])
        raw.info["bads"] = ["E0", "E3", "E4"]
        names = [raw.ch_names[i] for i in flatline._mask_eeg_picks(raw, "Cz")]
        self.assertEqual(flatline._global_flatline_segments(result, names), [])
        for cutoff in (-0.1, 1.1, np.nan):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                flatline._global_flatline_segments(result, names, cutoff)

    def test_configurations_cover_recording_use_full_raw_indices_and_omit_global_spans(self):
        raw = make_raw(n_times=50, auxiliary=True)
        result = prescribed_segments(raw, {
            "E0": [[4, 20]], "E1": [[10, 30]], "E2": [[12, 18]],
        })
        expected = [(0, 4, ()), (4, 10, (1,)), (10, 20, ()),
                    (20, 30, (2,)), (30, 50, ())]
        self.assertEqual(list(flatline._flatline_configurations(raw, result)), expected)
        self.assertEqual(list(flatline._flatline_configurations(raw, None)), [(0, 50, ())])


if __name__ == "__main__":
    unittest.main()
