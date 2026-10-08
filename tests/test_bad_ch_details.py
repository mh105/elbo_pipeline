import unittest

import mne
import numpy as np

from test_bad_ch_local_signals import make_raw as local_raw
from test_bad_ch_noise import bb_raw, std_raw, step2


class TestWindowDetails(unittest.TestCase):
    def assert_window_details(self, original, updated, flagged, details,
                              window_samples, retained, occurrence_cutoff=0.25):
        n_windows = len(retained)
        np.testing.assert_allclose(
            details["start_s"], np.arange(n_windows) * window_samples / original.info["sfreq"]
        )
        np.testing.assert_allclose(
            details["stop_s"], np.arange(1, n_windows + 1) * window_samples / original.info["sfreq"]
        )
        np.testing.assert_array_equal(details["retained"], retained)
        flags = details["violates"]
        self.assertEqual(flags.shape, (len(details["ch_names"]), n_windows))
        self.assertTrue(np.isnan(flags[:, ~np.asarray(retained)]).all())
        scored = flags[:, retained]
        self.assertTrue(np.isin(scored[~np.isnan(scored)], [0, 1]).all())
        expected = [name for name, row in zip(details["ch_names"], flags)
                    if np.any(row == 1) and np.nanmean(row) > occurrence_cutoff]
        self.assertEqual(flagged, expected)
        self.assertEqual(updated.info["bads"],
                         original.info["bads"] + expected)
        self.assertEqual(updated.annotations, original.annotations)
        np.testing.assert_array_equal(updated.get_data(), original.get_data())
        for i, name in enumerate(details["ch_names"]):
            windows = np.flatnonzero(flags[i] == 1)
            waveforms = details["flagged_data_uv"][name]
            self.assertEqual(waveforms.shape, (len(windows), window_samples))
            for waveform, window in zip(waveforms, windows):
                expected_waveform = original.get_data(
                    picks=[name], start=window * window_samples,
                    stop=(window + 1) * window_samples,
                )[0] * 1e6
                np.testing.assert_array_equal(waveform, expected_waveform)

    def test_std_robust_sds_channel_exclusions_and_annotated_windows(self):
        variances = np.ones((5, 9))
        variances[:2, 0] = 0.01
        variances[:, -1] = 100
        for window_samples in (100, 200):
            with self.subTest(window_samples=window_samples):
                raw = local_raw(variances, window_samples=window_samples,
                                first_samp=500, average=True)
                raw.crop(tmax=(4.5 * window_samples - 1) / raw.info["sfreq"])
                raw.info["bads"] = ["E5", "M5"]
                raw.set_channel_types({"E6": "eog", "M6": "eog"})
                raw.set_annotations(mne.Annotations([0.1], [0.1], ["BAD_excluded"]))
                original = raw.copy()
                kwargs = {"window_length": window_samples / raw.info["sfreq"]}
                expected = step2.bad_ch_by_std(raw.copy(), **kwargs)
                report, details = step2.bad_ch_by_std(raw, return_details=True, **kwargs)
                self.assertEqual(report, expected)
                self.assertEqual(report, {"std": ["E8", "M8"]})
                self.assert_window_details(original, raw, report["std"], details,
                                           window_samples, [False, True, True, True])
                names = details["ch_names"]
                self.assertEqual(names, [n for n in original.ch_names
                                         if n not in ("E5", "M5", "E6", "M6")])
                # make_raw's square waves have robust SD 1.4826 * sqrt(variance).
                np.testing.assert_allclose(details["robust_std_uv"][names.index("E0")],
                                           1.4826 * np.array([np.nan, 0.1, 1, 1]))
                np.testing.assert_allclose(details["robust_std_uv"][names.index("E8"), 1:],
                                           1.4826 * 10)
                np.testing.assert_array_equal(details["violates"][:, 1:],
                                              (details["robust_z"] > 5)[:, 1:])
                e8 = names.index("E8")
                np.testing.assert_array_equal(details["violates"][e8, 1:], 1)

    def test_flatline_global_exclusions_and_partial_tail(self):
        variances = np.ones((5, 7))
        variances[0, :3] = 0
        variances[1:3, 0] = 0
        variances[4, 0] = 0
        raw = local_raw(variances, first_samp=500)
        raw.crop(tmax=8.99)
        raw.info["bads"] = ["E5"]
        raw.set_channel_types({"E6": "eog"})
        raw.set_annotations(mne.Annotations([2.1], [0.1], ["BAD_excluded"]))
        # Annotates the global dropout; exact spans suit this short recording.
        flatlines = step2.detect_flatline_inplace(raw, downstream_highpass_hz=None)
        original = raw.copy()
        expected = step2.bad_ch_by_flatline(raw.copy(), flatlines=flatlines)
        report, details = step2.bad_ch_by_flatline(raw, flatlines=flatlines,
                                                   return_details=True)
        self.assertEqual(report, expected)
        self.assertEqual(report, {"flatline": ["E0"]})
        self.assert_window_details(original, raw, report["flatline"], details, 200,
                                   [False, False, True, True])
        self.assertEqual(details["ch_names"], [f"E{i}" for i in range(5)])
        np.testing.assert_array_equal(details["flat"][0], [np.nan, np.nan, 1, 0])
        np.testing.assert_array_equal(details["violates"][0], [np.nan, np.nan, 1, 0])

    def test_noise_details_preserve_mask_gaps_metrics_and_channel_order(self):
        for detector, factory in ((step2.bad_ch_by_std, std_raw),
                                  (step2.bad_ch_by_BBnoise, bb_raw)):
            for window_length in (1, 2):
                with self.subTest(detector=detector.__name__, window_length=window_length):
                    source = factory(window_length=window_length)
                    raw = mne.io.RawArray(source.get_data(), source.info.copy(),
                                          first_samp=750, verbose=False)
                    raw.info["bads"] = ["E1", "M1"]
                    raw.set_channel_types({"E2": "eog", "M2": "eog"})
                    samples = int(raw.info["sfreq"] * window_length)
                    raw.set_annotations(mne.Annotations([window_length + 0.1], [0.01],
                                                       ["BAD_existing"]))
                    for i in (0, 3, 4, 5):
                        raw._data[raw.ch_names.index(f"E{i}"), 2 * samples:2 * samples + 5] = 600e-6
                        raw._data[raw.ch_names.index(f"M{i}"), 2 * samples:2 * samples + 5] = -600e-6
                    original = raw.copy()
                    expected = detector(raw.copy(), window_length=window_length)
                    report, details = detector(raw, window_length=window_length, return_details=True)
                    self.assertEqual(report, expected)
                    criterion = next(iter(report))
                    self.assertTrue(report[criterion])
                    self.assert_window_details(original, raw, report[criterion], details,
                                               samples, [True, False, False, True])
                    self.assertEqual(details["ch_names"], [n for n in original.ch_names
                                                         if n not in ("E1", "M1", "E2", "M2")])
                    if detector is step2.bad_ch_by_std:
                        for metric in ("robust_std_uv", "robust_z"):
                            self.assertEqual(details[metric].shape, details["violates"].shape)
                            self.assertTrue(np.isnan(details[metric][:, [1, 2]]).all())
                        i = details["ch_names"].index("E8")
                        # std_raw's square waves have robust SD 1.4826 * amplitude.
                        np.testing.assert_allclose(details["robust_std_uv"][i, [0, 3]],
                                                   1.4826 * np.array([100, 10.7]))
                        np.testing.assert_array_equal(details["violates"][:, [0, 3]],
                                                      details["robust_z"][:, [0, 3]] > 5)
                    else:
                        self.assertEqual(details["bands_hz"], ((0, 4), (4, 8), (8, 12), (12, 30), (30, 45)))
                        for metric in ("band_mean_psd_db", "robust_z", "band_violates"):
                            self.assertEqual(details[metric].shape, (*details["violates"].shape, 5))
                            self.assertTrue(np.isnan(details[metric][:, [1, 2]]).all())
                        band_flags = details["band_violates"][:, [0, 3]]
                        np.testing.assert_array_equal(band_flags, details["robust_z"][:, [0, 3]] > 5)
                        np.testing.assert_array_equal(details["n_bad_bands"][:, [0, 3]], band_flags.sum(axis=2))
                        np.testing.assert_array_equal(details["violates"][:, [0, 3]], band_flags.sum(axis=2) >= 3)

    def test_fully_masked_details_raise_without_mutating_raw(self):
        for detector, factory in ((step2.bad_ch_by_std, std_raw),
                                  (step2.bad_ch_by_BBnoise, bb_raw)):
            with self.subTest(detector=detector.__name__):
                raw = factory()
                raw.set_annotations(mne.Annotations([0], [8], ["BAD_all"]))
                original = raw.copy()
                with self.assertRaisesRegex(ValueError, "No complete unmasked windows"):
                    detector(raw, return_details=True)
                self.assertEqual(raw.info["bads"], [])
                self.assertEqual(raw.annotations, original.annotations)
                np.testing.assert_array_equal(raw.get_data(), original.get_data())


if __name__ == "__main__":
    unittest.main()
