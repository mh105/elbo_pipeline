import sys
import unittest
from pathlib import Path

import mne
import numpy as np


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import step2_bad_ch_time_detection as step2


DETECTORS = (
    step2.bad_ch_by_smallness,
    step2.bad_ch_by_std,
    step2.bad_ch_by_BBnoise,
    step2.bad_ch_by_voltage,
    step2.bad_ch_by_HFnoise,
)


def make_raw(average=True, with_bads=False):
    rng = np.random.default_rng(73)
    data = rng.normal(scale=10e-6, size=(10, 1000))
    data[8] = 0  # Native reference before re-referencing.
    names = [f"E{i}" for i in range(8)] + ["Cz", "EOG"]
    info = mne.create_info(names, 250, ["eeg"] * 9 + ["eog"])
    raw = mne.io.RawArray(data, info, verbose=False)
    if with_bads:
        raw.info["bads"] = ["E0"]
        raw._data[0] = 1e-3
    if average:
        raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw


class TestAverageReference(unittest.TestCase):
    def test_statistical_detectors_reject_native_reference_even_when_named(self):
        for detector in DETECTORS[:3]:
            for reference_channel in (None, "Cz"):
                with self.subTest(detector=detector.__name__, reference=reference_channel):
                    raw = make_raw(average=False)
                    data = raw.get_data().copy()
                    self.assertEqual(step2._resolve_reference_channel(raw, None), "Cz")
                    with self.assertRaisesRegex(ValueError, "common-average referenced"):
                        detector(raw, reference_channel=reference_channel)
                    self.assertEqual(raw.info["bads"], [])
                    self.assertEqual(len(raw.annotations), 0)
                    np.testing.assert_array_equal(raw.get_data(), data)

    def test_voltage_hf_and_flatline_accept_native_reference(self):
        for detector in DETECTORS[3:] + (step2.bad_ch_by_flatline,):
            for reference_channel in (None, "Cz"):
                with self.subTest(detector=detector.__name__, reference=reference_channel):
                    raw = make_raw(average=False)
                    result = detector(raw, reference_channel=reference_channel)
                    criterion = detector.__name__.removeprefix("bad_ch_by_")
                    self.assertEqual(list(result), [criterion])
                    self.assertNotIn("Cz", result[criterion])
                    self.assertEqual(raw.info["bads"], result[criterion])

    def test_average_only_selection_includes_zero_channel(self):
        raw = make_raw()
        raw._data[8] = 0
        raw._data[:8] -= raw._data[:8].mean(axis=0)
        picks, _, _ = step2._prepare_channel_windows(
            raw, "Cz", 2, 0.25, require_average=True
        )
        self.assertIn(8, picks)
        self.assertEqual(len(picks), 9)

    def test_average_only_mask_does_not_exclude_zero_channel(self):
        raw = make_raw().pick(["E0", "E1", "E2", "Cz"])
        raw._data[:] = np.array([1, -1, 0, 0])[:, None] * 1e-6
        raw._data[2] = 1e-6
        raw._data[0] -= 1e-6
        raw._data[:3, 0] = np.array([600, -300, -300]) * 1e-6
        # One of all four channels is exactly 25%; excluding Cz would mask it.
        step2.mask_segment_by_voltage(raw, reference_channel="Cz", require_average=True)
        self.assertEqual(len(raw.annotations), 0)

    def test_all_five_accept_actual_mne_average_reference(self):
        for detector in DETECTORS:
            for reference_channel in (None, "Cz"):
                with self.subTest(detector=detector.__name__, reference=reference_channel):
                    raw = make_raw(with_bads=True)
                    raw.set_annotations(mne.Annotations([1], [0], ["event"]))
                    data = raw.get_data().copy()
                    annotations = raw.annotations.copy()
                    self.assertIsNone(step2._resolve_reference_channel(raw, reference_channel))
                    result = detector(raw, reference_channel=reference_channel)
                    criterion = detector.__name__.removeprefix("bad_ch_by_")
                    self.assertEqual(list(result), [criterion])
                    self.assertEqual(raw.info["bads"], ["E0"] + result[criterion])
                    self.assertNotIn("E0", result[criterion])
                    self.assertNotIn("EOG", result[criterion])
                    np.testing.assert_array_equal(raw.get_data(), data)
                    self.assertEqual(raw.annotations, annotations)

    def test_all_five_reject_nonzero_sample_mean_without_native_reference(self):
        for detector in DETECTORS:
            with self.subTest(detector=detector.__name__):
                raw = make_raw()
                raw._data[:9] += np.tile([-1e-6, 1e-6], 500)
                # The overall mean is zero, but the mean at each sample is not.
                self.assertAlmostEqual(raw.get_data(picks="eeg").mean(), 0)
                original = raw.get_data().copy()
                with self.assertRaisesRegex(ValueError, "common-average referenced"):
                    detector(raw)
                self.assertEqual(raw.info["bads"], [])
                np.testing.assert_array_equal(raw.get_data(), original)

    def test_check_excludes_bads_and_non_eeg_channels(self):
        raw = make_raw(with_bads=True)
        raw._data[-1] = np.inf
        eeg = mne.pick_types(raw.info, eeg=True, exclude=[])
        self.assertGreater(np.max(np.abs(raw.get_data(picks=eeg).mean(axis=0))), 1e-6)
        step2._check_average_reference(raw)
        self.assertIsNone(step2._resolve_reference_channel(raw, None))

    def test_tolerance_and_single_sample_violation_even_inside_bad_annotation(self):
        raw = make_raw()
        raw._data[:9] += 0.5e-10
        step2._check_average_reference(raw)
        raw._data[:9, 500] += 2e-10
        raw.set_annotations(mne.Annotations([2], [0.1], ["BAD_test"]))
        with self.assertRaisesRegex(ValueError, "sample-wise"):
            step2._resolve_reference_channel(raw, None)

    def test_changed_bads_require_rechecking_current_average(self):
        raw = make_raw()
        step2._check_average_reference(raw)
        raw.info["bads"] = ["E1"]
        with self.assertRaisesRegex(ValueError, "current non-bad EEG"):
            step2._resolve_reference_channel(raw, None)
        raw.set_eeg_reference("average", projection=False, verbose=False)
        self.assertIsNone(step2._resolve_reference_channel(raw, None))

    def test_old_reference_name_is_not_excluded_after_average_reference(self):
        raw = make_raw()
        picks, _, _ = step2._prepare_channel_windows(raw, "Cz", 2, 0.25)
        self.assertIn(raw.ch_names.index("Cz"), picks)
        self.assertEqual(len(picks), 9)

    def test_unapplied_average_reference_projector_does_not_satisfy_check(self):
        raw = make_raw(average=False).drop_channels(["Cz"])
        raw.set_eeg_reference("average", projection=True, verbose=False)
        with self.assertRaisesRegex(ValueError, "common-average referenced"):
            step2._resolve_reference_channel(raw, None)
        self.assertFalse(raw.info["projs"][0]["active"])

    def test_flatline_still_requires_zero_reference(self):
        for reference_channel in (None, "Cz"):
            with self.subTest(reference=reference_channel):
                raw = make_raw()
                with self.assertRaisesRegex(ValueError, "exactly one entirely zero"):
                    step2.bad_ch_by_flatline(raw, reference_channel=reference_channel)
                self.assertEqual(raw.info["bads"], [])

    def test_multiple_zero_channels_can_fall_back_to_average_reference(self):
        raw = make_raw()
        raw._data[:2] = 0
        raw._data[2:9] -= raw._data[2:9].mean(axis=0)
        self.assertIsNone(step2._resolve_reference_channel(raw, None))
        with self.assertRaisesRegex(ValueError, "exactly one entirely zero"):
            step2.bad_ch_by_flatline(raw)

    def test_mask_accepts_average_reference_without_mutating_signal(self):
        raw = make_raw()
        raw._data[:9, 100:110] = np.array([600, -600, 600, -600, 1, -1, 1, -1, 0])[:, None] * 1e-6
        original = raw.get_data().copy()
        step2.mask_segment_by_voltage(raw)
        self.assertIn("BAD_voltage_mask", raw.annotations.description)
        np.testing.assert_array_equal(raw.get_data(), original)


if __name__ == "__main__":
    unittest.main()
