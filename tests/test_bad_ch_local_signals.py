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
from step2_bad_ch_time_detection import (
    _reference_channel,
    bad_ch_by_flatline,
    bad_ch_by_nan,
    bad_ch_by_smallness,
    detect_flatline_inplace,
)


def flatline(raw, **kwargs):
    """Detect flatlines (masking global dropouts on raw), then score them.

    The window rules here concern exact spans, so local intervals are not
    widened for a later high-pass (5.5 s would cover these short recordings).
    """
    flatlines = detect_flatline_inplace(raw, kwargs.get("reference_channel"),
                                        downstream_highpass_hz=None)
    return bad_ch_by_flatline(raw, flatlines=flatlines, **kwargs)


def make_raw(variances, window_samples=200, first_samp=0, average=False):
    """Make prescribed variances with a native reference or balanced channel pairs."""
    variances = np.asarray(variances, dtype=float)
    n_windows, n_channels = variances.shape
    wave = np.tile([-1.0, 1.0], window_samples // 2)
    data = (2 + np.sqrt(variances.T)[..., None] * wave) * 1e-6
    data = data.reshape(n_channels, n_windows * window_samples)
    names = [f"E{i}" for i in range(n_channels)]
    if average:
        data = np.vstack([data, -data])
        names += [f"M{i}" for i in range(n_channels)]
    else:
        data = np.vstack([data, np.zeros(data.shape[1])])
        names += ["Cz"]
    info = mne.create_info(names, 100, "eeg")
    return mne.io.RawArray(data, info, first_samp=first_samp, verbose=False)


class TestBadChByNan(unittest.TestCase):
    def test_single_invalid_samples_any_channel_and_annotation(self):
        raw = make_raw(np.ones((1, 5)))
        raw.set_channel_types({"E4": "eog"})
        raw._data[0, 0] = np.nan
        raw._data[1, -1] = np.inf
        raw._data[2, 50] = -np.inf
        raw._data[4, 50] = np.nan
        raw.set_annotations(mne.Annotations([0.4], [0.2], ["BAD_test"]))
        raw.info["bads"] = ["E3", "E0"]
        original = raw.get_data().copy()
        result = bad_ch_by_nan(raw)
        self.assertEqual(result, {"nan": ["E0", "E1", "E2", "E4"]})
        self.assertEqual(bad_ch_by_nan(raw), result)
        self.assertEqual(raw.info["bads"], ["E3", "E0", "E1", "E2", "E4"])
        np.testing.assert_array_equal(raw.get_data(), original)

    def test_finite_data_do_not_add_bads(self):
        raw = make_raw(np.ones((1, 2)))
        raw.info["bads"] = ["E1"]
        self.assertEqual(bad_ch_by_nan(raw), {"nan": []})
        self.assertEqual(raw.info["bads"], ["E1"])


class TestNativeReferenceChannel(unittest.TestCase):
    def test_auto_reference_ignores_bads_and_auxiliary_channels(self):
        raw = make_raw(np.ones((1, 3)))
        raw._data[0:2] = 0
        raw.set_channel_types({"E0": "eog"})
        raw.info["bads"] = ["E1"]
        self.assertEqual(_reference_channel(raw, "native"), "Cz")
        raw.info["bads"] = ["E1", "Cz"]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            _reference_channel(raw, "native")

    def test_missing_or_multiple_candidates_raise(self):
        for candidates in (0, 2):
            with self.subTest(candidates=candidates):
                raw = make_raw(np.ones((1, 3)))
                if candidates == 0:
                    raw._data[-1] = 1e-6
                else:
                    raw._data[0] = 0
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    _reference_channel(raw, "native")

    def test_explicit_reference_checks_only_that_channel(self):
        raw = make_raw(np.ones((1, 3)))
        raw._data[0] = 0
        raw.info["bads"] = ["Cz"]
        with patch.object(step2, "_zero_eeg_channels",
                          wraps=step2._zero_eeg_channels) as zeros:
            self.assertEqual(_reference_channel(raw, "native", "Cz"), "Cz")
        zeros.assert_called_once_with(raw, [3], None)
        with self.assertRaisesRegex(ValueError, "not entirely zero"):
            _reference_channel(raw, "native", "E1")
        raw.set_channel_types({"E0": "eog"})
        for name in ("missing", "E0"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "EEG channel"):
                _reference_channel(raw, "native", name)


class TestBadChBySmallness(unittest.TestCase):
    def test_upper_outlier_removal_and_strict_occurrence_cutoff(self):
        for low_windows, expected in ((1, []), (2, ["E0", "M0"])):
            with self.subTest(low_windows=low_windows):
                variances = np.ones((4, 9))
                variances[:low_windows, 0] = 0.01
                variances[:, -1] = 100
                raw = make_raw(variances, average=True)
                original = raw.get_data().copy()
                self.assertEqual(bad_ch_by_smallness(raw), {"smallness": expected})
                self.assertEqual(raw.info["bads"], expected)
                np.testing.assert_array_equal(raw.get_data(), original)

    def test_custom_parameters(self):
        # E0 has a log-variance robust z of about -2.7 in the first two windows.
        variances = np.tile([1.0, 0.9, 1.1, 0.95, 1.05, 1.0, 0.85, 1.15, 100], (4, 1))
        variances[:2, 0] = 0.65
        for kwargs, expected in (
            ({"window_length": 1.0}, ["E0", "M0"]),
            ({"window_length": 1.0, "occurrence_cutoff": 0.5}, []),
            ({"window_length": 1.0, "robust_z_cutoff": 3.0}, []),
        ):
            with self.subTest(kwargs=kwargs):
                raw = make_raw(variances, window_samples=100, average=True)
                self.assertEqual(bad_ch_by_smallness(raw, **kwargs), {"smallness": expected})
                self.assertEqual(raw.info["bads"], expected)

    def test_low_gain_channel_is_found_among_right_skewed_variances(self):
        # Variances spread over a factor of 20 across channels, as in real recordings.
        # On the variance scale, median - 2 * 1.4826 * MAD is below zero, so no channel
        # could ever be small; on the log scale a low-gain channel stands out.
        variances = np.exp(np.tile(np.linspace(-1.5, 1.5, 30), (4, 1)))
        mad = np.median(np.abs(variances[0] - np.median(variances[0])))
        self.assertLess(np.median(variances[0]) - 2 * 1.4826 * mad, 0)
        self.assertEqual(bad_ch_by_smallness(make_raw(variances, average=True)),
                         {"smallness": []})
        variances[:, 0] = np.exp(-2.6)  # About a quarter of the typical amplitude.
        self.assertEqual(bad_ch_by_smallness(make_raw(variances, average=True)),
                         {"smallness": ["E0", "M0"]})

    def test_strong_artifact_channels_do_not_inflate_baseline(self):
        # A mean/std baseline keeps the three 100s and its lower cutoff goes negative.
        variances = np.tile([0.6, 0.9, 1.1, 0.95, 1.05, 1.0, 100, 100, 100], (2, 1))
        raw = make_raw(variances, average=True)
        self.assertEqual(bad_ch_by_smallness(raw), {"smallness": ["E0", "M0"]})

    def test_bads_auxiliary_and_reference_are_excluded(self):
        variances = np.ones((2, 11))
        variances[:, 0] = 0.01
        raw = make_raw(variances, average=True)
        raw.info["bads"] = ["E9"]
        raw._data[9] = np.nan
        raw.set_channel_types({"E10": "eog"})
        raw._data[10] = np.inf
        raw.set_eeg_reference("average", projection=False, verbose=False)
        result = bad_ch_by_smallness(raw)
        self.assertEqual(result, {"smallness": ["E0", "M0"]})
        raw.set_eeg_reference("average", projection=False, verbose=False)
        self.assertEqual(bad_ch_by_smallness(raw), {"smallness": []})
        self.assertEqual(raw.info["bads"], ["E9", "E0", "M0"])


class TestBadChByFlatline(unittest.TestCase):
    def test_constant_nonzero_and_strict_occurrence_cutoff(self):
        for flat_windows, expected in ((1, []), (2, ["E0"])):
            with self.subTest(flat_windows=flat_windows):
                variances = np.ones((4, 5))
                variances[:flat_windows, 0] = 0
                raw = make_raw(variances)
                flatlines = detect_flatline_inplace(raw, downstream_highpass_hz=None)
                bridged = raw.get_data().copy()  # Detection bridged E0's flat span.
                self.assertEqual(bad_ch_by_flatline(raw, flatlines=flatlines),
                                 {"flatline": expected})
                self.assertEqual(raw.info["bads"], expected)
                np.testing.assert_array_equal(raw.get_data(), bridged)

    def test_peer_flatness_does_not_suppress_channel_flags(self):
        # Up to 5 of 24 channels flat at once stay below the 25% global cutoff.
        for flat_channels in (2, 3, 5):
            with self.subTest(flat_channels=flat_channels):
                expected = [f"E{i}" for i in range(flat_channels)]
                variances = np.ones((4, 24))
                variances[:, :flat_channels] = 0
                raw = make_raw(variances)
                self.assertEqual(flatline(raw), {"flatline": expected})
                self.assertEqual(raw.info["bads"], expected)
                self.assertEqual(len(raw.annotations), 0)

    def test_global_dropout_windows_leave_occurrence_denominator(self):
        variances = np.ones((4, 5))
        variances[:3] = 0
        variances[3, 0] = 0
        raw = make_raw(variances)
        self.assertEqual(flatline(raw), {"flatline": ["E0"]})
        self.assertEqual(raw.info["bads"], ["E0"])
        np.testing.assert_array_equal(raw.annotations.description, ["BAD_by_flatline"])

    def test_bads_and_auxiliary_are_excluded_from_channel_flags(self):
        raw = make_raw([[0, 0, 0] + [1] * 13])  # 3 of 14 eligible channels are flat.
        raw.info["bads"] = ["E14"]
        raw.set_channel_types({"E15": "eog"})
        self.assertEqual(flatline(raw), {"flatline": ["E0", "E1", "E2"]})
        self.assertEqual(raw.info["bads"], ["E14", "E0", "E1", "E2"])

    def test_flatlines_are_required(self):
        raw = make_raw(np.ones((1, 5)))
        with self.assertRaisesRegex(ValueError, "detect_flatline_inplace"):
            bad_ch_by_flatline(raw, flatlines=None)
        with self.assertRaises(TypeError):
            bad_ch_by_flatline(raw)

    def test_custom_parameters_and_tiny_variation_is_flat(self):
        variances = np.ones((4, 9))  # Two flat channels of nine are not a dropout.
        variances[:2, 0] = 0
        for cutoff, expected in ((0.25, ["E0", "E1"]), (0.5, ["E1"])):
            with self.subTest(cutoff=cutoff):
                raw = make_raw(variances, window_samples=100)
                raw._data[1] = 2e-6
                raw._data[1, ::100] += 1e-15
                self.assertEqual(
                    flatline(raw, window_length=1.0, occurrence_cutoff=cutoff),
                    {"flatline": expected},
                )
                self.assertEqual(raw.info["bads"], expected)


def run_detector(detector, raw, **kwargs):
    """Call smallness directly; detect flatlines first for bad_ch_by_flatline."""
    if detector is bad_ch_by_flatline:
        return flatline(raw, **kwargs)
    return detector(raw, **kwargs)


class TestWindowRules(unittest.TestCase):
    def test_partial_tail_and_bad_annotations_excluded_with_raw_start_respected(self):
        for detector in (bad_ch_by_smallness, bad_ch_by_flatline):
            with self.subTest(detector=detector.__name__):
                variances = np.ones((5, 9))
                variances[0, 0] = 0
                variances[4, 0] = 0
                average = detector is bad_ch_by_smallness
                raw = make_raw(variances, first_samp=500, average=average)
                raw.crop(tmax=8.99)  # Four complete windows and one 1-second tail.
                kwargs = {}
                if not average:  # Detect once, as the pipeline does, then score.
                    kwargs["flatlines"] = detect_flatline_inplace(
                        raw, downstream_highpass_hz=None
                    )
                raw.set_annotations(mne.Annotations([0], [2], ["BAD_test"]))
                criterion = detector.__name__.removeprefix("bad_ch_by_")
                # Only one of four full windows violates the rule.
                self.assertEqual(detector(raw, **kwargs), {criterion: []})
                self.assertEqual(raw.info["bads"], [])
                self.assertEqual(detector(raw, occurrence_cutoff=0.2, **kwargs),
                                 {criterion: []})
                raw.set_annotations(None)
                expected = ["E0", "M0"] if average else ["E0"]
                self.assertEqual(detector(raw, occurrence_cutoff=0.2, **kwargs),
                                 {criterion: expected})
                self.assertEqual(raw.info["bads"], expected)

    def test_invalid_values_need_nan_detection_first(self):
        for detector in (bad_ch_by_smallness, bad_ch_by_flatline):
            with self.subTest(detector=detector.__name__):
                raw = make_raw(np.ones((1, 9)), average=detector is bad_ch_by_smallness)
                raw._data[0, 50] = np.nan
                with self.assertRaisesRegex(ValueError, "bad_ch_by_nan"):
                    run_detector(detector, raw)
                self.assertEqual(raw.info["bads"], [])
                if detector is bad_ch_by_smallness:
                    raw.info["bads"] = ["E0"]  # bad_ch_by_nan needs the native reference.
                else:
                    self.assertEqual(bad_ch_by_nan(raw), {"nan": ["E0"]})
                criterion = detector.__name__.removeprefix("bad_ch_by_")
                self.assertEqual(run_detector(detector, raw), {criterion: []})
                self.assertEqual(raw.info["bads"], ["E0"])

    def test_invalid_parameters_and_insufficient_data(self):
        for detector in (bad_ch_by_smallness, bad_ch_by_flatline):
            for kwargs in (
                {"window_length": 0}, {"window_length": np.inf},
                {"window_length": 0.01}, {"window_length": 3},
                {"occurrence_cutoff": -0.1}, {"occurrence_cutoff": 1.1},
                {"occurrence_cutoff": np.nan},
            ):
                with self.subTest(detector=detector.__name__, kwargs=kwargs):
                    raw = make_raw(np.ones((1, 2)))
                    with self.assertRaises(ValueError):
                        run_detector(detector, raw, **kwargs)
                    self.assertEqual(raw.info["bads"], [])
            with self.subTest(detector=detector.__name__), self.assertRaisesRegex(
                ValueError, "At least two"
            ):
                raw = make_raw([[1]], average=detector is bad_ch_by_smallness)
                if detector is bad_ch_by_smallness:
                    raw.info["bads"] = ["M0"]
                run_detector(detector, raw)
        for cutoff in (0, -1, np.nan, np.inf):
            with self.subTest(cutoff=cutoff), self.assertRaisesRegex(
                ValueError, "robust_z_cutoff"
            ):
                bad_ch_by_smallness(make_raw([[1, 1]]), robust_z_cutoff=cutoff)


if __name__ == "__main__":
    unittest.main()
