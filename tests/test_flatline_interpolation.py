import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
from mne.channels.interpolation import _make_interpolation_matrix


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import step2_bad_ch_time_detection as step2


NAMES = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7",
         "C3", "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4",
         "P8", "PO3", "PO4", "O1", "Oz", "O2"]


def make_raw(n_times=3000):  # 12 s, so a 2-s window survives the test dropouts.
    rng = np.random.default_rng(76)
    data = rng.normal(scale=10e-6, size=(len(NAMES) + 1, n_times))
    data[:len(NAMES)] -= data[NAMES.index("Cz")]
    raw = mne.io.RawArray(data, mne.create_info(NAMES + ["EOG"], 250,
                                              ["eeg"] * len(NAMES) + ["eog"]), verbose=False)
    raw.set_montage(mne.channels.make_standard_montage("colin27_1020"), verbose=False)
    return raw


def flatlines_for(raw, spans, channel_fraction_cutoff=0.25):
    """Prescribed flat spans, with the global dropouts that detect_flatline_inplace
    would decide from the current bads (without annotating raw)."""
    names = list(spans)
    segments = [np.array(spans[name], dtype=int).reshape(-1, 2) for name in names]
    empty = [np.empty((0, 2), dtype=int) for _ in names]
    result = {"ch_names": names, "segments": segments,
              "segments_on_subtypes": {
                  "segments_on_raw": segments, "segments_on_highpass": empty,
                  "segments_on_ramp": empty, "segments_on_curve": empty,
              },
              "eeg_ch_names": [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])],
              "raw_ch_names": list(raw.ch_names), "sfreq": raw.info["sfreq"],
              "first_samp": raw.first_samp, "n_times": raw.n_times, "reference_channel": "Cz",
              "min_samples": 10, "tolerance_v": 1e-9, "highpass_hz": 1.0}
    step2._mask_segment_by_flatline(raw.copy(), result, channel_fraction_cutoff)
    return result


def mne_reconstruction(raw, bads):
    expected = raw.copy()
    eeg_names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]
    expected.info["bads"] = [name for name in bads if name in eeg_names]
    expected.interpolate_bads(reset_bads=False, on_bad_position="raise", verbose=False)
    return expected


class TestSphericalSpline(unittest.TestCase):
    def test_weights_match_mne_for_original_indices_and_float64(self):
        raw = make_raw()
        geometry = step2._spline_geometry(raw)
        targets = np.array([raw.ch_names.index("F3"), raw.ch_names.index("O1")])
        donors = geometry["picks"][~np.isin(geometry["picks"], targets)]
        weights = step2._spline_weights(geometry, donors, targets)
        expected = _make_interpolation_matrix(geometry["positions"][geometry["row_of"][donors]],
                                              geometry["positions"][geometry["row_of"][targets]])
        np.testing.assert_allclose(weights, expected, rtol=1e-8, atol=1e-10)
        self.assertEqual(weights.dtype, np.dtype(np.float64))
        self.assertEqual(geometry["kernel"].dtype, np.dtype(np.float64))
        self.assertIs(weights, step2._spline_weights(geometry, donors, targets))
        self.assertEqual(geometry["row_of"][-1], -1)

    def test_whole_bad_nan_channel_mean_and_repair_match_mne(self):
        raw = make_raw()
        raw._data[NAMES.index("T7")] = np.nan
        raw.info["bads"] = ["T7", "EOG"]
        original = raw.copy()
        expected = mne_reconstruction(original, raw.info["bads"])
        geometry = step2._spline_geometry(raw)
        mean = step2._robust_mean(raw, raw.info["bads"], geometry=geometry)
        np.testing.assert_array_equal(raw.get_data(), original.get_data())
        np.testing.assert_allclose(mean, expected._data[geometry["picks"]].mean(axis=0),
                                   rtol=0, atol=1e-15)
        step2._interpolate_bads_and_flatlines_inplace(raw, raw.info["bads"], geometry=geometry)
        np.testing.assert_allclose(raw.get_data(), expected.get_data(), rtol=0, atol=1e-15)
        self.assertEqual(raw.info["bads"], ["T7", "EOG"])

    def test_overlapping_local_spans_use_genuine_donors_and_skip_global_bad(self):
        raw = make_raw()
        raw.info["bads"] = ["T7", "EOG"]
        raw._data[NAMES.index("T7")] = np.nan
        raw._data[NAMES.index("F7"), 200:600] = 13e-6
        raw._data[NAMES.index("F3"), 400:800] = -7e-6
        raw.set_annotations(mne.Annotations([450 / 250], [100 / 250], ["BAD_dropout"]))
        spans = {"F7": [[200, 600]], "F3": [[400, 800]]}
        for name in ["F4", "F8", "FC5", "FC1", "FC2", "FC6"]:
            spans[name] = [[450, 550]]
            raw._data[NAMES.index(name), 450:550] = 0
        flatlines = flatlines_for(raw, spans)
        original = raw.copy()
        expected = mne_reconstruction(original, ["T7"])
        intervals = [(200, 400, ["F7"]), (400, 450, ["F7", "F3"]),
                     (550, 600, ["F7", "F3"]), (600, 800, ["F3"])]
        for start, stop, local in intervals:
            repaired = mne_reconstruction(original, ["T7"] + local)
            rows = [raw.ch_names.index(name) for name in ["T7"] + local]
            expected._data[rows, start:stop] = repaired._data[rows, start:stop]
        geometry = step2._spline_geometry(raw)
        mean = step2._robust_mean(raw, raw.info["bads"], flatlines, geometry)
        self.assertTrue(np.isfinite(mean).all())
        np.testing.assert_array_equal(raw.get_data(), original.get_data())
        np.testing.assert_allclose(mean, expected._data[geometry["picks"]].mean(axis=0),
                                   rtol=0, atol=1e-15)
        step2._interpolate_bads_and_flatlines_inplace(raw, raw.info["bads"], flatlines, geometry)
        np.testing.assert_allclose(raw.get_data(), expected.get_data(), rtol=0, atol=1e-15)
        raw._data[geometry["picks"]] -= mean
        np.testing.assert_allclose(raw._data[geometry["picks"]].mean(axis=0), 0,
                                   rtol=0, atol=1e-18)
        np.testing.assert_array_equal(raw._data[NAMES.index("Cz")], -mean)
        np.testing.assert_array_equal(raw._data[-1], original._data[-1])
        self.assertEqual(raw.info["bads"], ["T7", "EOG"])
        self.assertEqual(raw.annotations, original.annotations)
        for donors, targets in geometry["weights"]:
            self.assertFalse(set(donors) & set(targets))
            self.assertNotIn(NAMES.index("T7"), donors)

    def test_repeated_mean_and_final_repair_reuse_the_kernel_and_weights(self):
        raw = make_raw()
        flatlines = flatlines_for(raw, {"F7": [[200, 600]], "F3": [[400, 800]]})
        with patch.object(step2, "_calc_g", wraps=step2._calc_g) as kernel:
            geometry = step2._spline_geometry(raw)
            with patch.object(step2, "solve", wraps=step2.solve) as solve:
                mean = step2._robust_mean(raw, ["T7"], flatlines, geometry)
                first = solve.call_count
                self.assertGreater(first, 0)
                again = step2._robust_mean(raw, ["T7"], flatlines, geometry)
                step2._interpolate_bads_and_flatlines_inplace(raw, ["T7"], flatlines, geometry)
                self.assertEqual(solve.call_count, first)
            self.assertEqual(kernel.call_count, 1)
        np.testing.assert_array_equal(mean, again)

    def test_custom_global_cutoff_repairs_spans_the_default_considers_dropout(self):
        raw = make_raw()
        names = ["F7", "F3", "F4", "F8", "FC5", "FC1", "FC2", "FC6"]
        rows = [raw.ch_names.index(name) for name in names]
        raw._data[rows, 450:550] = 0
        flatlines = flatlines_for(raw, {name: [[450, 550]] for name in names})
        self.assertEqual(flatlines["global_segments"], [[450, 550]])  # 8 of 29 > 25%.
        default = raw.copy()
        step2._interpolate_bads_and_flatlines_inplace(default, [], flatlines)
        np.testing.assert_array_equal(default.get_data(), raw.get_data())

        flatlines = flatlines_for(raw, {name: [[450, 550]] for name in names},
                                  channel_fraction_cutoff=0.5)
        self.assertEqual(flatlines["global_segments"], [])
        expected = raw.copy()
        repaired = mne_reconstruction(raw, names)
        expected._data[rows, 450:550] = repaired._data[rows, 450:550]
        geometry = step2._spline_geometry(raw)
        mean = step2._robust_mean(raw, [], flatlines, geometry)
        np.testing.assert_allclose(mean, expected._data[geometry["picks"]].mean(axis=0),
                                   rtol=0, atol=1e-15)
        step2._interpolate_bads_and_flatlines_inplace(raw, [], flatlines, geometry)
        np.testing.assert_allclose(raw.get_data(), expected.get_data(), rtol=0, atol=1e-15)
        self.assertGreater(np.abs(raw._data[rows, 450:550]).max(), 0)

    def test_local_only_repair_does_not_need_whole_channel_bads(self):
        raw = make_raw()
        row = raw.ch_names.index("F7")
        raw._data[row, 100:300] = 0
        flatlines = flatlines_for(raw, {"F7": [[100, 300]]})
        expected = raw.copy()
        repaired = mne_reconstruction(raw, ["F7"])
        expected._data[row, 100:300] = repaired._data[row, 100:300]
        geometry = step2._spline_geometry(raw)
        mean = step2._robust_mean(raw, [], flatlines, geometry)
        step2._interpolate_bads_and_flatlines_inplace(raw, [], flatlines, geometry)
        np.testing.assert_allclose(mean, expected._data[geometry["picks"]].mean(axis=0),
                                   rtol=0, atol=1e-15)
        np.testing.assert_allclose(raw.get_data(), expected.get_data(), rtol=0, atol=1e-15)

    def test_long_recording_is_applied_in_chunks(self):
        raw = make_raw(n_times=17000)
        raw._data[NAMES.index("T7")] = np.nan
        expected = mne_reconstruction(raw, ["T7"])
        geometry = step2._spline_geometry(raw)
        mean = step2._robust_mean(raw, ["T7"], geometry=geometry)
        step2._interpolate_bads_and_flatlines_inplace(raw, ["T7"], geometry=geometry)
        np.testing.assert_allclose(mean, expected._data[geometry["picks"]].mean(axis=0),
                                   rtol=0, atol=1e-15)
        np.testing.assert_allclose(raw.get_data(), expected.get_data(), rtol=0, atol=1e-15)

    def test_invalid_sensor_position_and_missing_donors_raise(self):
        raw = make_raw()
        raw.info["chs"][NAMES.index("F3")]["loc"][:3] = np.nan
        with self.assertRaisesRegex(ValueError, "valid sensor positions.*F3"):
            step2._spline_geometry(raw)
        geometry = step2._spline_geometry(make_raw())
        with self.assertRaisesRegex(ValueError, "genuine EEG donor"):
            step2._spline_weights(geometry, [], [0])


if __name__ == "__main__":
    unittest.main()
