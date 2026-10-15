import sys
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import step2_bad_ch_time_detection as step2
from test_bad_ch_local_signals import set_sphere_positions


# bad_ch_by_RANSAC also requires average reference; its tests need a realistic
# montage, so they live in test_bad_ch_cross_channel.py.
AVERAGE_ONLY = (step2.bad_ch_by_std, step2.bad_ch_by_BBnoise, step2.bad_ch_by_correlation)
EITHER = (step2.bad_ch_by_voltage, step2.bad_ch_by_HFnoise)
NATIVE_ONLY = (step2.bad_ch_by_nan, step2.bad_ch_by_flatline)


def make_raw(average=True, with_bads=False):
    rng = np.random.default_rng(73)
    data = rng.normal(scale=10e-6, size=(10, 1000))
    data[8] = 0  # Native reference before re-referencing.
    names = [f"E{i}" for i in range(8)] + ["Cz", "EOG"]
    info = mne.create_info(names, 250, ["eeg"] * 9 + ["eog"])
    raw = set_sphere_positions(mne.io.RawArray(data, info, verbose=False))
    if with_bads:
        raw.info["bads"] = ["E0"]
        raw._data[0] = 1e-3
    if average:
        raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw


def criterion(method):
    return method.__name__.removeprefix("bad_ch_by_")


class TestReferenceRequirements(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_native_only_methods_accept_auto_or_named_reference(self):
        for method in NATIVE_ONLY:
            for reference_channel in (None, "Cz"):
                with self.subTest(method=method.__name__, reference=reference_channel):
                    raw = make_raw(average=False)
                    kwargs = {"reference_channel": reference_channel}
                    if method is step2.bad_ch_by_flatline:
                        kwargs["flatlines"] = step2.detect_flatline_inplace(
                            raw, reference_channel
                        )
                        step2.mask_segment_by_flatline(raw, kwargs["flatlines"])
                    result = method(raw, **kwargs)
                    self.assertEqual(result, {criterion(method): []})

    def test_flatline_accepts_one_eligible_channel_and_less_than_two_seconds(self):
        raw = make_raw(average=False).pick(["E0", "Cz"])
        raw.crop(tmax=0.396)  # 100 samples at 250 Hz.
        flatlines = step2.detect_flatline_inplace(raw)
        step2.mask_segment_by_flatline(raw, flatlines)
        result, details = step2.bad_ch_by_flatline(
            raw, flatlines=flatlines, return_details=True,
        )
        self.assertEqual(result, {"flatline": []})
        self.assertEqual(details["ch_names"], ["E0"])
        np.testing.assert_array_equal(details["flat_fraction"], [0])

    def test_native_only_methods_reject_re_referenced_data(self):
        methods = NATIVE_ONLY + (step2.detect_flatline_inplace, step2.pca_cleaning_inplace,
                                 step2.bad_ch_under_robust_ref)
        for method in methods:
            for reference_channel, message in ((None, "exactly one"), ("Cz", "not entirely zero")):
                with self.subTest(method=method.__name__, reference=reference_channel):
                    raw = make_raw()
                    original = raw.get_data().copy()
                    kwargs = {"reference_channel": reference_channel}
                    if method is step2.bad_ch_by_flatline:
                        native = make_raw(average=False)
                        kwargs["flatlines"] = step2.detect_flatline_inplace(
                            native, reference_channel
                        )
                        step2.mask_segment_by_flatline(native, kwargs["flatlines"])
                    with self.assertRaisesRegex(ValueError, message):
                        method(raw, **kwargs)
                    self.assertEqual(raw.info["bads"], [])
                    self.assertEqual(len(raw.annotations), 0)
                    np.testing.assert_array_equal(raw.get_data(), original)

    def test_named_reference_lets_flatline_flag_another_zero_channel(self):
        raw = make_raw(average=False)
        raw._data[0] = 0
        with self.assertRaisesRegex(ValueError, "exactly one"):
            step2.detect_flatline_inplace(raw)
        self.assertEqual(step2.bad_ch_by_nan(raw, reference_channel="Cz"), {"nan": []})
        flatlines = step2.detect_flatline_inplace(raw, reference_channel="Cz")
        step2.mask_segment_by_flatline(raw, flatlines)
        result = step2.bad_ch_by_flatline(raw, reference_channel="Cz", flatlines=flatlines)
        self.assertEqual(result, {"flatline": ["E0"]})
        # With E0 now bad, the automatic search finds Cz again.
        self.assertEqual(step2.bad_ch_by_flatline(raw, flatlines=flatlines), {"flatline": []})

    def test_named_reference_may_be_bad_but_auto_search_skips_bads(self):
        raw = make_raw(average=False)
        raw.info["bads"] = ["Cz"]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            step2.detect_flatline_inplace(raw)
        flatlines = step2.detect_flatline_inplace(raw, reference_channel="Cz")
        step2.mask_segment_by_flatline(raw, flatlines)
        self.assertEqual(step2.bad_ch_by_flatline(raw, reference_channel="Cz",
                                                  flatlines=flatlines),
                         {"flatline": []})

    def test_nan_requires_an_entirely_zero_reference(self):
        raw = make_raw(average=False)
        raw._data[8, 10] = np.nan
        with self.assertRaisesRegex(ValueError, "exactly one"):
            step2.bad_ch_by_nan(raw)

    def test_average_only_detectors_reject_native_reference_even_if_bad(self):
        for detector in AVERAGE_ONLY:
            for bads in ([], ["Cz"]):
                with self.subTest(detector=detector.__name__, bads=bads):
                    raw = make_raw(average=False)
                    raw.info["bads"] = list(bads)
                    original = raw.get_data().copy()
                    with self.assertRaisesRegex(ValueError, r"\['Cz'\] are entirely zero"):
                        detector(raw)
                    self.assertEqual(raw.info["bads"], bads)
                    self.assertEqual(len(raw.annotations), 0)
                    np.testing.assert_array_equal(raw.get_data(), original)

    def test_average_only_detectors_accept_any_re_reference(self):
        for detector in AVERAGE_ONLY:
            for shifted in (False, True):
                with self.subTest(detector=detector.__name__, shifted=shifted):
                    raw = make_raw(with_bads=True)
                    if shifted:  # A nonzero sample-wise mean no longer fails the check.
                        raw._data[:9] += np.tile([-1e-6, 1e-6], 500)
                    raw.set_annotations(mne.Annotations([1], [0], ["event"]))
                    data = raw.get_data().copy()
                    annotations = raw.annotations.copy()
                    result = detector(raw)
                    self.assertEqual(list(result), [criterion(detector)])
                    flagged = list(dict.fromkeys(sum(result.values(), [])))
                    self.assertEqual(raw.info["bads"], ["E0"] + flagged)
                    self.assertNotIn("E0", flagged)
                    self.assertNotIn("EOG", flagged)
                    np.testing.assert_array_equal(raw.get_data(), data)
                    self.assertEqual(raw.annotations, annotations)

    def test_average_only_selection_includes_former_reference(self):
        picks, _, _ = step2._prepare_channel_windows(make_raw(), 2, 0.25, "average")
        self.assertIn(8, picks)
        self.assertEqual(len(picks), 9)

    def test_either_detectors_exclude_only_a_native_reference(self):
        for average, n_picks in ((False, 8), (True, 9)):
            with self.subTest(average=average):
                raw = make_raw(average=average)
                picks, _, _ = step2._prepare_channel_windows(raw, 2, 0.25, "either")
                self.assertEqual(len(picks), n_picks)
                self.assertEqual(8 in picks, average)
                for detector in EITHER:
                    self.assertEqual(detector(raw.copy()), {criterion(detector): []})

    def test_voltage_plot_can_be_suppressed(self):
        for plot, n_figures in ((False, 0), (True, 1)):
            with self.subTest(plot=plot):
                plt.close("all")
                step2.bad_ch_by_voltage(make_raw(), plot=plot)
                self.assertEqual(len(plt.get_fignums()), n_figures)

    def test_either_detectors_ignore_zero_bads_but_reject_several_zero_channels(self):
        for detector in EITHER:
            with self.subTest(detector=detector.__name__):
                raw = make_raw(average=False)
                raw._data[0] = 0
                with self.assertRaisesRegex(ValueError, "bad_ch_by_flatline"):
                    detector(raw)
                raw.info["bads"] = ["E0"]
                self.assertEqual(detector(raw), {criterion(detector): []})

    def test_mask_excludes_native_reference_unless_average_is_required(self):
        raw = make_raw(average=False).pick(["E0", "E1", "E2", "Cz"])
        raw._data[:3] = 1e-6
        raw._data[0, 0] = 600e-6
        # One of three channels exceeds; counting Cz would make it exactly 25%.
        step2.mask_segment_by_voltage(raw)
        self.assertEqual(list(raw.annotations.description), ["BAD_voltage_mask"])
        with self.assertRaisesRegex(ValueError, "entirely zero"):
            step2.mask_segment_by_voltage(raw, require_average=True)

    def test_mask_accepts_average_reference_without_mutating_signal(self):
        raw = make_raw()
        raw._data[:9, 100:110] = np.array([600, -600, 600, -600, 1, -1, 1, -1, 0])[:, None] * 1e-6
        original = raw.get_data().copy()
        step2.mask_segment_by_voltage(raw, require_average=True)
        self.assertIn("BAD_voltage_mask", raw.annotations.description)
        np.testing.assert_array_equal(raw.get_data(), original)


if __name__ == "__main__":
    unittest.main()
