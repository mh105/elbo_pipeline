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
    would decide and annotate from the current bads."""
    names = list(spans)
    segments = [np.array(spans[name], dtype=int).reshape(-1, 2) for name in names]
    empty = [np.empty((0, 2), dtype=int) for _ in names]
    result = {"ch_names": names, "segments_exact": segments, "segments_bridged": segments,
              "segments_on_subtypes": {
                  "segments_on_raw": segments, "segments_on_highpass": empty,
                  "segments_on_ramp": empty,
              },
              "eeg_ch_names": [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])],
              "raw_ch_names": list(raw.ch_names), "sfreq": raw.info["sfreq"],
              "first_samp": raw.first_samp, "n_times": raw.n_times, "reference_channel": "Cz",
              "min_samples": 10, "tolerance_v": 1e-9, "highpass_hz": 1.0}
    step2._mask_segment_by_flatline(raw, result, channel_fraction_cutoff)
    return result


def mne_reconstruction(raw, bads):
    expected = raw.copy()
    eeg_names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]
    expected.info["bads"] = [name for name in bads if name in eeg_names]
    expected.interpolate_bads(reset_bads=False, on_bad_position="raise", verbose=False)
    return expected


def faded(original, repaired, start, stop, sfreq=250):
    """One channel's samples with the repair blended in by 0.25-s raised cosines
    at both interval ends."""
    weight = np.ones(stop - start)
    length = min(int(round(0.25 * sfreq)), (stop - start) // 2)
    rise = 0.5 - 0.5 * np.cos(np.pi * (np.arange(length) + 0.5) / length)
    weight[:length] = rise
    weight[stop - start - length:] = rise[::-1]
    return weight * repaired[start:stop] + (1 - weight) * original[start:stop]


def blended_reconstruction(raw, bads, parts):
    """Full MNE reconstructions blended over explicitly specified test intervals."""
    def reconstruct(local):
        invalid = sorted(set(bads).union(local))
        return mne_reconstruction(raw, invalid) if invalid else raw.copy()

    expected = reconstruct([])
    for start, stop, before, after in parts:
        current = reconstruct(after)
        if before == after:
            expected._data[:, start:stop] = current._data[:, start:stop]
        else:
            previous = reconstruct(before)
            length = stop - start
            weight = 0.5 - 0.5 * np.cos(np.pi * (np.arange(length) + 0.5) / length)
            expected._data[:, start:stop] = (
                (1 - weight) * previous._data[:, start:stop]
                + weight * current._data[:, start:stop]
            )
    return expected


def overlapping_parts(global_dropout=False):
    """Expected 250-Hz transitions for F7 [200, 600), F3 [400, 800)."""
    middle = (
        [(400, 425, ["F7"], ["F7", "F3"]),
         (425, 450, ["F7", "F3"], []),
         (550, 575, [], ["F7", "F3"]),
         (575, 600, ["F7", "F3"], ["F3"])]
        if global_dropout else
        [(400, 462, ["F7"], ["F7", "F3"]),
         (462, 538, ["F7", "F3"], ["F7", "F3"]),
         (538, 600, ["F7", "F3"], ["F3"])]
    )
    return [(200, 262, [], ["F7"]), (262, 400, ["F7"], ["F7"]),
            *middle, (600, 738, ["F3"], ["F3"]), (738, 800, ["F3"], [])]


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

    def test_whole_bad_nan_channel_mean_matches_mne(self):
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
        reconstruction = blended_reconstruction(original, ["T7"], overlapping_parts(True))
        expected = reconstruction.copy()
        expected._data[NAMES.index("T7")] = original._data[NAMES.index("T7")]
        geometry = step2._spline_geometry(raw)
        mean = step2._robust_mean(raw, raw.info["bads"], flatlines, geometry)
        self.assertTrue(np.isfinite(mean).all())
        np.testing.assert_array_equal(raw.get_data(), original.get_data())
        np.testing.assert_allclose(mean, reconstruction._data[geometry["picks"]].mean(axis=0),
                                   rtol=0, atol=1e-15)
        with patch.object(step2, "_spline_weights", wraps=step2._spline_weights) as spline:
            step2.interpolate_local_flatline_inplace(raw, flatlines)
        np.testing.assert_allclose(raw.get_data(), expected.get_data(), rtol=0, atol=1e-15)
        np.testing.assert_array_equal(raw._data[-1], original._data[-1])
        self.assertEqual(raw.info["bads"], ["T7", "EOG"])
        self.assertEqual(raw.annotations, original.annotations)
        self.assertGreater(spline.call_count, 0)
        for call in spline.call_args_list:
            donors, targets = call.args[1], call.args[2]
            self.assertFalse(set(donors) & set(targets))
            self.assertNotIn(NAMES.index("T7"), set(donors) | set(targets))

    def test_recording_edge_spans_fade_from_temporally_filled_samples(self):
        for start, stop in [(0, 400), (2600, 3000), (0, 40), (2960, 3000)]:
            with self.subTest(start=start, stop=stop):
                raw = make_raw()
                row = raw.ch_names.index("F7")
                step2._bridge_flat_spans_inplace(raw._data[row], [[start, stop]])
                flatlines = flatlines_for(raw, {"F7": [[start, stop]]})
                original = raw.copy()
                repaired = mne_reconstruction(original, ["F7"])
                expected = original.copy()
                expected._data[row, start:stop] = faded(
                    original._data[row], repaired._data[row], start, stop
                )
                mean = step2._robust_mean(raw, [], flatlines)
                np.testing.assert_allclose(mean, expected._data[:len(NAMES)].mean(axis=0),
                                           rtol=0, atol=1e-15)
                step2.interpolate_local_flatline_inplace(raw, flatlines)
                np.testing.assert_allclose(raw.get_data(), expected.get_data(),
                                           rtol=0, atol=1e-15)

    def test_overlaps_share_faded_reconstructions_and_reuse_valid_donors(self):
        for bads in ([], ["T7"]):
            with self.subTest(bads=bads):
                raw = make_raw()
                raw.info["bads"] = bads
                if bads:
                    raw._data[NAMES.index("T7")] = np.nan
                flatlines = flatlines_for(raw, {"F7": [[200, 600]], "F3": [[400, 800]]})
                original = raw.copy()
                expected = blended_reconstruction(original, bads, overlapping_parts())
                mean = step2._robust_mean(raw, bads, flatlines)
                np.testing.assert_allclose(mean, expected._data[:len(NAMES)].mean(axis=0),
                                           rtol=0, atol=1e-15)
                np.testing.assert_array_equal(raw.get_data(), original.get_data())
                step2.interpolate_local_flatline_inplace(raw, flatlines)
                for name in ("F7", "F3"):
                    row = NAMES.index(name)
                    np.testing.assert_allclose(raw._data[row], expected._data[row],
                                               rtol=0, atol=1e-15)
                reconstructed = mne_reconstruction(raw, bads) if bads else raw
                np.testing.assert_allclose(mean, reconstructed._data[:len(NAMES)].mean(axis=0),
                                           rtol=0, atol=1e-15)

    def test_simultaneous_changes_pass_through_common_flat_channels(self):
        for persistent in (False, True):
            with self.subTest(persistent=persistent):
                raw = make_raw()
                spans = {"F7": [[200, 600]], "F3": [[600, 1000]]}
                common = ["Fz"] if persistent else []
                if persistent:
                    spans["Fz"] = [[100, 1100]]
                flatlines = flatlines_for(raw, spans)
                original = raw.copy()
                parts = [(200, 262, common, common + ["F7"]),
                         (262, 538, common + ["F7"], common + ["F7"]),
                         (538, 600, common + ["F7"], common),
                         (600, 662, common, common + ["F3"]),
                         (662, 938, common + ["F3"], common + ["F3"]),
                         (938, 1000, common + ["F3"], common)]
                if persistent:
                    parts = [(100, 150, [], common), (150, 200, common, common),
                             *parts, (1000, 1050, common, common), (1050, 1100, common, [])]
                expected = blended_reconstruction(original, [], parts)
                mean = step2._robust_mean(raw, [], flatlines)
                step2.interpolate_local_flatline_inplace(raw, flatlines)
                np.testing.assert_allclose(raw.get_data(), expected.get_data(),
                                           rtol=0, atol=1e-15)
                np.testing.assert_allclose(mean, raw._data[:len(NAMES)].mean(axis=0),
                                           rtol=0, atol=1e-15)
                np.testing.assert_array_equal(raw._data[NAMES.index("F7"), 600:],
                                              original._data[NAMES.index("F7"), 600:])
                np.testing.assert_array_equal(raw._data[NAMES.index("F3"), :600],
                                              original._data[NAMES.index("F3"), :600])

    def test_short_configuration_caps_both_fades_at_half_its_duration(self):
        raw = make_raw()
        flatlines = flatlines_for(raw, {"F7": [[200, 1000]], "F3": [[500, 512]]})
        expected = blended_reconstruction(raw, [], [
            (200, 262, [], ["F7"]), (262, 500, ["F7"], ["F7"]),
            (500, 506, ["F7"], ["F7", "F3"]),
            (506, 512, ["F7", "F3"], ["F7"]),
            (512, 938, ["F7"], ["F7"]), (938, 1000, ["F7"], []),
        ])
        mean = step2._robust_mean(raw, [], flatlines)
        step2.interpolate_local_flatline_inplace(raw, flatlines)
        np.testing.assert_allclose(raw.get_data(), expected.get_data(), rtol=0, atol=1e-15)
        np.testing.assert_allclose(mean, expected._data[:len(NAMES)].mean(axis=0),
                                   rtol=0, atol=1e-15)

    def test_one_sample_configuration_and_channel_order_preserve_the_plan(self):
        raw = make_raw()
        flatlines = flatlines_for(raw, {"F7": [[500, 600]], "F3": [[501, 700]]})
        original = raw.copy()
        expected = blended_reconstruction(raw, [], [
            (500, 501, ["F7"], ["F7"]),
            (501, 550, ["F7"], ["F7", "F3"]),
            (550, 551, ["F7", "F3"], ["F7", "F3"]),
            (551, 600, ["F7", "F3"], ["F3"]),
            (600, 650, ["F3"], ["F3"]), (650, 700, ["F3"], []),
        ])
        mean = step2._robust_mean(raw, [], flatlines)
        step2.interpolate_local_flatline_inplace(raw, flatlines)
        np.testing.assert_allclose(raw.get_data(), expected.get_data(), rtol=0, atol=1e-15)
        np.testing.assert_allclose(mean, expected._data[:len(NAMES)].mean(axis=0),
                                   rtol=0, atol=1e-15)
        reordered = flatlines.copy()
        for key in ("ch_names", "segments_exact", "segments_bridged"):
            reordered[key] = list(reversed(flatlines[key]))
        np.testing.assert_array_equal(step2._robust_mean(original, [], reordered), mean)
        step2.interpolate_local_flatline_inplace(original, reordered)
        np.testing.assert_array_equal(original.get_data(), raw.get_data())

    def test_local_repair_leaves_bad_channels_alone(self):
        raw = make_raw()
        raw._data[NAMES.index("F7"), 100:300] = 0
        raw._data[NAMES.index("F3"), 200:400] = 0
        flatlines = flatlines_for(raw, {"F7": [[100, 300]], "F3": [[200, 400]]})
        raw.info["bads"] = ["F7", "EOG"]  # Flagged after flatline detection.
        expected = raw.copy()
        repaired = mne_reconstruction(raw, ["F7", "F3"])
        row = raw.ch_names.index("F3")
        expected._data[row, 200:400] = faded(raw._data[row], repaired._data[row], 200, 400)
        step2.interpolate_local_flatline_inplace(raw, flatlines)
        np.testing.assert_allclose(raw.get_data(), expected.get_data(), rtol=0, atol=1e-15)
        self.assertEqual(raw.info["bads"], ["F7", "EOG"])

    def test_repeated_mean_reuses_the_kernel_and_weights(self):
        raw = make_raw()
        flatlines = flatlines_for(raw, {"F7": [[200, 600]], "F3": [[400, 800]]})
        with patch.object(step2, "_calc_g", wraps=step2._calc_g) as kernel:
            geometry = step2._spline_geometry(raw)
            with patch.object(step2, "solve", wraps=step2.solve) as solve:
                mean = step2._robust_mean(raw, ["T7"], flatlines, geometry)
                first = solve.call_count
                self.assertGreater(first, 0)
                again = step2._robust_mean(raw, ["T7"], flatlines, geometry)
                self.assertEqual(solve.call_count, first)
            self.assertEqual(kernel.call_count, 1)
        np.testing.assert_array_equal(mean, again)

    def test_custom_global_cutoff_repairs_spans_the_default_considers_dropout(self):
        raw = make_raw()
        names = ["F7", "F3", "F4", "F8", "FC5", "FC1", "FC2", "FC6"]
        rows = [raw.ch_names.index(name) for name in names]
        raw._data[rows, 450:550] = 0
        unannotated = raw.copy()
        flatlines = flatlines_for(raw, {name: [[450, 550]] for name in names})
        self.assertEqual(flatlines["global_segments_exact"], [[450, 550]])  # 8 of 29 > 25%.
        default = raw.copy()
        step2.interpolate_local_flatline_inplace(default, flatlines)
        np.testing.assert_array_equal(default.get_data(), raw.get_data())

        raw = unannotated
        flatlines = flatlines_for(raw, {name: [[450, 550]] for name in names},
                                  channel_fraction_cutoff=0.5)
        self.assertEqual(flatlines["global_segments_exact"], [])
        expected = raw.copy()
        repaired = mne_reconstruction(raw, names)
        for row in rows:
            expected._data[row, 450:550] = faded(raw._data[row], repaired._data[row], 450, 550)
        geometry = step2._spline_geometry(raw)
        mean = step2._robust_mean(raw, [], flatlines, geometry)
        np.testing.assert_allclose(mean, expected._data[geometry["picks"]].mean(axis=0),
                                   rtol=0, atol=1e-15)
        step2.interpolate_local_flatline_inplace(raw, flatlines)
        np.testing.assert_allclose(raw.get_data(), expected.get_data(), rtol=0, atol=1e-15)
        self.assertGreater(np.abs(raw._data[rows, 450:550]).max(), 0)

    def test_local_only_repair_does_not_need_whole_channel_bads(self):
        raw = make_raw()
        row = raw.ch_names.index("F7")
        raw._data[row, 100:300] = 0
        flatlines = flatlines_for(raw, {"F7": [[100, 300]]})
        expected = raw.copy()
        repaired = mne_reconstruction(raw, ["F7"])
        expected._data[row, 100:300] = faded(raw._data[row], repaired._data[row], 100, 300)
        geometry = step2._spline_geometry(raw)
        mean = step2._robust_mean(raw, [], flatlines, geometry)
        np.testing.assert_allclose(mean, expected._data[geometry["picks"]].mean(axis=0),
                                   rtol=0, atol=1e-15)
        step2.interpolate_local_flatline_inplace(raw, flatlines)
        np.testing.assert_allclose(raw.get_data(), expected.get_data(), rtol=0, atol=1e-15)

    def test_requires_the_native_reference(self):
        raw = make_raw()
        row = raw.ch_names.index("F7")
        raw._data[row, 100:300] = 0
        flatlines = flatlines_for(raw, {"F7": [[100, 300]]})
        eeg = slice(0, len(NAMES))
        average, moved = raw.copy(), raw.copy()
        average._data[eeg] -= average._data[eeg].mean(axis=0)
        moved._data[eeg] -= moved._data[NAMES.index("FC1")].copy()
        # The default is the saved native reference Cz, no longer zero in either.
        for name, data in [("average", average), ("moved", moved)]:
            with self.subTest(name=name):
                original = data.get_data()
                with self.assertRaisesRegex(ValueError, "not entirely zero"):
                    step2.interpolate_local_flatline_inplace(data, flatlines)
                np.testing.assert_array_equal(data.get_data(), original)
        step2.interpolate_local_flatline_inplace(moved, flatlines, reference_channel="FC1")
        np.testing.assert_array_equal(moved._data[raw.ch_names.index("FC1")], 0)
        # Before validation moves it, a bad native reference still qualifies.
        raw.info["bads"] = ["Cz"]
        step2.interpolate_local_flatline_inplace(raw, flatlines)
        self.assertGreater(np.abs(raw._data[row, 150:250]).min(), 0)
        np.testing.assert_array_equal(raw._data[raw.ch_names.index("Cz")], 0)

    def test_long_recording_is_applied_in_chunks(self):
        raw = make_raw(n_times=17000)
        raw._data[NAMES.index("T7")] = np.nan
        raw.info["bads"] = ["T7"]
        expected = mne_reconstruction(raw, ["T7"])
        geometry = step2._spline_geometry(raw)
        mean = step2._robust_mean(raw, ["T7"], geometry=geometry)
        np.testing.assert_allclose(mean, expected._data[geometry["picks"]].mean(axis=0),
                                   rtol=0, atol=1e-15)
        row = raw.ch_names.index("F7")
        raw._data[row, 100:16900] = 0  # Longer than one 8192-sample chunk.
        flatlines = flatlines_for(raw, {"F7": [[100, 16900]]})
        expected = raw.copy()
        repaired = mne_reconstruction(raw, ["T7", "F7"])
        expected._data[row, 100:16900] = faded(raw._data[row], repaired._data[row], 100, 16900)
        mean = step2._robust_mean(raw, ["T7"], flatlines)
        step2.interpolate_local_flatline_inplace(raw, flatlines)
        np.testing.assert_allclose(raw.get_data(), expected.get_data(), rtol=0, atol=1e-15)
        reconstructed = mne_reconstruction(raw, ["T7"])
        np.testing.assert_allclose(mean, reconstructed._data[geometry["picks"]].mean(axis=0),
                                   rtol=0, atol=1e-15)

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
