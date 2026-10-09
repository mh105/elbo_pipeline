import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
from scipy.signal import butter, lfilter, sosfiltfilt


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import step2_bad_ch_time_detection as step2
from mne_denoise.asr import _calibration
from mne_denoise.asr._types import ASRState


NAMES = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "C3", "Cz", "C4", "O1", "O2"]


def make_raw(seconds=75, sfreq=250, first_samp=1000, reference="Cz"):
    rng = np.random.default_rng(1028)
    data = rng.normal(scale=5e-6, size=(len(NAMES) + 1, int(seconds * sfreq)))
    data[:len(NAMES)] -= data[NAMES.index(reference)]
    raw = mne.io.RawArray(
        data, mne.create_info(NAMES + ["EOG"], sfreq,
                              ["eeg"] * len(NAMES) + ["eog"]),
        first_samp=first_samp, verbose=False,
    )
    raw.set_montage(mne.channels.make_standard_montage("colin27_1020"), verbose=False)
    return raw


def flatlines_for(raw, spans, reference="Cz"):
    """Supply the already widened local intervals saved by flatline detection."""
    return {
        "ch_names": list(spans),
        "segments": [np.asarray(value, dtype=int).reshape(-1, 2)
                     for value in spans.values()],
        "raw_ch_names": list(raw.ch_names), "eeg_ch_names": NAMES.copy(),
        "sfreq": float(raw.info["sfreq"]), "first_samp": int(raw.first_samp),
        "n_times": int(raw.n_times), "reference_channel": reference,
        "global_segments": [], "widen_samples": 20,
    }


class FakeASR:
    """Control ASR decisions while retaining its actual state/data interfaces."""

    def __init__(self, retained=None):
        self.retained = retained
        self.calibration_data = None
        self.processing_data = None
        self.calibration_kwargs = None
        self.processing_state = None
        self.corrected = None

    def calibrate(self, X, sfreq, **kwargs):
        self.calibration_data = X.copy()
        self.calibration_kwargs = kwargs
        n_channels, n_times = X.shape
        retained = np.ones(n_times, dtype=bool)
        if self.retained is not None:
            retained[:] = False
            retained[:self.retained] = True
        else:
            retained[10:15] = False
        b, a = _calibration._design_statistics_filter(sfreq, kwargs["filter_kind"])
        state = ASRState(
            M=np.eye(n_channels), T=np.eye(n_channels),
            thresholds=np.ones(n_channels), calibration_patterns=np.eye(n_channels),
            filter_b=b, filter_a=a, cov=np.eye(n_channels), rank=n_channels,
            filter_zi=np.zeros((n_channels, max(len(a), len(b)) - 1)),
        )
        return state, {"clean_sample_mask": retained,
                       "calibration_samples": int(retained.sum()),
                       "blocksize": kwargs["blocksize"],
                       "thresholds": state.thresholds.copy(),
                       "threshold_mu": np.full(n_channels, 0.2),
                       "threshold_sigma": np.full(n_channels, 0.04)}

    def process(self, X, sfreq, state, **kwargs):
        self.processing_data = X.copy()
        self.processing_state = state
        offsets = (np.arange(X.shape[0]) + 1)[:, None] * 2e-6
        self.corrected = X + offsets
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[100:250] = True
        mask[300:450] = True
        return self.corrected.copy(), {
            "window_starts": np.array([0, 100, 150, 200, 300, 400]),
            "window_stops": np.array([100, 150, 200, 300, 400, X.shape[1]]),
            "n_components_reconstructed": np.array([0, 2, 1, 0, 3, 0]),
            "sample_mask": mask,
            "component_variances": np.zeros((6, X.shape[0])),
            "component_thresholds": np.ones((6, X.shape[0])),
        }


class TestASRWrapper(unittest.TestCase):
    def run_fake(self, raw, fake, **kwargs):
        kwargs.setdefault("flatlines", flatlines_for(raw, {}))
        with patch.object(step2, "calibrate_asr", side_effect=fake.calibrate), \
                patch.object(step2, "process_asr", side_effect=fake.process):
            return step2.apply_asr_inplace(raw, **kwargs)

    def test_calibration_exclusions_mapping_and_full_timeline_correction(self):
        raw = make_raw(reference="F3")
        raw.info["bads"] = ["F7", "Cz"]
        raw.set_annotations(mne.Annotations([1, 3, 5], [1, 1, 0.1],
                                            ["BAD_manual", "bad_acquisition", "event"]))
        flatlines = flatlines_for(raw, {
            "Fp1": [[1250, 1400]], "F3": [[1750, 1850]],
            "F7": [[0, raw.n_times]],
        })
        original = raw.get_data()
        annotations = raw.annotations.copy()
        rejected = step2._rejected_samples(raw)
        rejected[1250:1400] = True
        rejected[1750:1850] = True
        expected_indices = np.flatnonzero(~rejected)
        selected = [i for i, name in enumerate(raw.ch_names)
                    if name in NAMES and name not in ["F7", "Cz", "F3"]]
        fake = FakeASR()
        report, details = self.run_fake(
            raw, fake, reference_channel="F3", flatlines=flatlines, return_details=True,
        )

        np.testing.assert_array_equal(fake.calibration_data,
                                      original[np.ix_(selected, expected_indices)])
        np.testing.assert_array_equal(details["calibration_input_sample_indices"],
                                      expected_indices)
        retained = np.ones(len(expected_indices), dtype=bool)
        retained[10:15] = False
        np.testing.assert_array_equal(details["calibration_sample_indices"],
                                      expected_indices[retained])
        np.testing.assert_array_equal(fake.processing_data, original[selected])
        self.assertEqual(fake.calibration_kwargs["calibration"], "auto")
        self.assertEqual(fake.calibration_kwargs["cutoff"], 20)
        self.assertEqual(fake.calibration_kwargs["blocksize"], 25)
        self.assertEqual(details["ch_names"], [raw.ch_names[i] for i in selected])
        self.assertEqual(details["reference_channel"], "F3")
        # Global exclusions affect calibration only: correction reaches them too.
        np.testing.assert_array_equal(raw._data[selected, 250:500],
                                      fake.corrected[:, 250:500])
        for name in ["F7", "Cz", "F3", "EOG"]:
            row = raw.ch_names.index(name)
            np.testing.assert_array_equal(raw._data[row], original[row])
        self.assertEqual(raw.annotations, annotations)
        self.assertEqual(raw.info["bads"], ["F7", "Cz"])
        self.assertEqual(report["component_segments"], [(100, 200), (300, 400)])
        self.assertEqual(report["corrected_segments"], [(100, 250), (300, 450)])
        for segments in report.values():
            self.assertTrue(all(type(start) is int and type(stop) is int
                                for start, stop in segments))

    def test_voltage_mask_preserves_existing_annotations_and_uses_final_good_channels(self):
        raw = make_raw()
        raw.info["bads"] = ["F7"]
        # A newly excluded channel cannot contribute to the final denominator.
        raw._data[raw.ch_names.index("F7"), 2500:2525] = 1e-3
        rows = [raw.ch_names.index(name) for name in ["Fp1", "Fp2", "F3"]]
        raw._data[rows, 5000:5025] = 1e-3
        raw.set_annotations(mne.Annotations(
            [10, 12, 14], [1, 1, 0.25],
            ["BAD_voltage_mask", "bad_voltage_mask", "BAD_manual"],
        ))
        fake = FakeASR()
        _, details = self.run_fake(raw, fake, reference_channel="Cz", return_details=True)
        descriptions = list(raw.annotations.description)
        self.assertEqual(descriptions.count("BAD_voltage_mask"), 2)
        self.assertIn("bad_voltage_mask", descriptions)
        self.assertIn("BAD_manual", descriptions)
        annotation = next(index for index, name in enumerate(descriptions)
                          if name == "BAD_voltage_mask"
                          and np.isclose(raw.annotations.onset[index] - raw.first_time, 20))
        self.assertAlmostEqual(raw.annotations.onset[annotation] - raw.first_time, 20)
        self.assertAlmostEqual(raw.annotations.duration[annotation], 0.1)
        indices = details["calibration_input_sample_indices"]
        self.assertNotIn(2500, indices)  # Existing exact-name mask remains excluded.
        self.assertNotIn(3000, indices)  # Existing lowercase BAD* remains excluded.
        self.assertFalse(np.isin(np.arange(5000, 5025), indices).any())

    def test_minimum_is_checked_after_automatic_selection_without_voltage_writeback(self):
        raw = make_raw()
        original = raw.get_data()
        fake = FakeASR(retained=60 * 250 - 1)
        designer = _calibration._design_statistics_filter
        with self.assertRaisesRegex(ValueError, "60|minute"):
            self.run_fake(raw, fake, reference_channel="Cz")
        self.assertIsNone(fake.processing_data)
        np.testing.assert_array_equal(raw.get_data(), original)
        self.assertIs(_calibration._design_statistics_filter, designer)

    def test_insufficient_input_calibration_support_does_not_run_asr(self):
        raw = make_raw(seconds=65)
        raw.set_annotations(mne.Annotations([0], [6], ["BAD_manual"]))
        original = raw.get_data()
        fake = FakeASR()
        with self.assertRaisesRegex(ValueError, "60|minute"):
            self.run_fake(raw, fake, reference_channel="Cz")
        self.assertIsNone(fake.calibration_data)
        np.testing.assert_array_equal(raw.get_data(), original)

    def test_native_auto_selection_hits_minimum_guard_before_calibration_filtering(self):
        raw = make_raw()
        start = 40 * 250
        t = np.arange(raw.n_times - start) / 250
        # Two of ten channels exceed native auto's one-channel tolerance while
        # remaining below the global-mask fraction.
        rows = [raw.ch_names.index(name) for name in ["Fp1", "Fp2"]]
        raw._data[rows, start:] += (
            np.linspace(100e-6, 400e-6, len(t)) * np.sin(2 * np.pi * 12 * t)
        )
        original = raw.get_data()
        retained_masks = []
        native_mask = _calibration._create_sample_mask_from_windows
        native_filter = _calibration._lfilter_channels
        native_designer = _calibration._design_statistics_filter

        def capture_mask(*args, **kwargs):
            retained = native_mask(*args, **kwargs)
            retained_masks.append(retained.copy())
            return retained

        with patch.object(_calibration, "_create_sample_mask_from_windows",
                          side_effect=capture_mask), \
                patch.object(step2, "_asr_filter_channels",
                             wraps=step2._asr_filter_channels) as filtering, \
                patch.object(step2, "process_asr") as processing:
            with self.assertRaisesRegex(ValueError, "60|minute"):
                step2.apply_asr_inplace(raw, "Cz", flatlines=flatlines_for(raw, {}))
        self.assertEqual(len(retained_masks), 1)
        self.assertEqual(len(retained_masks[0]), raw.n_times)
        self.assertLess(retained_masks[0].sum(), 60 * 250)
        filtering.assert_not_called()
        processing.assert_not_called()
        np.testing.assert_array_equal(raw.get_data(), original)
        self.assertIs(_calibration._lfilter_channels, native_filter)
        self.assertIs(_calibration._design_statistics_filter, native_designer)

    def test_default_return_is_report_and_blocksize_tracks_sampling_rate(self):
        for sfreq in [100, 250, 1000]:
            with self.subTest(sfreq=sfreq):
                raw = make_raw(sfreq=sfreq)
                fake = FakeASR()
                result = self.run_fake(raw, fake, reference_channel="Cz")
                self.assertIsInstance(result, dict)
                self.assertEqual(set(result), {"component_segments", "corrected_segments"})
                self.assertEqual(fake.calibration_kwargs["blocksize"], int(0.1 * sfreq))

    def test_sqrt_eigenvalue_cutoffs_follow_amplitude_and_reach_processing(self):
        raw = make_raw()
        fake = FakeASR()
        calibrate = fake.calibrate

        def calibrate_with_spectrum(X, sfreq, **kwargs):
            state, diagnostics = calibrate(X, sfreq, **kwargs)
            n_channels = X.shape[0]
            eigenvalues = np.array([1.] + [4.] * (n_channels - 2) + [16.])
            patterns = np.eye(n_channels)
            patterns[np.ix_([0, n_channels - 1], [0, n_channels - 1])] = [
                [0.6, -0.8], [0.8, 0.6],
            ]
            return replace(
                state, cov=(patterns * eigenvalues) @ patterns.T,
                M=(patterns * np.sqrt(eigenvalues)) @ patterns.T,
                calibration_patterns=patterns, T=patterns.T,
            ), diagnostics

        fake.calibrate = calibrate_with_spectrum
        with patch.object(step2, "plot_asr_calibration") as plotting, \
                patch.object(step2, "plot_asr_effect"):
            _, details = self.run_fake(raw, fake, reference_channel="Cz", return_details=True)
        n_channels = len(details["ch_names"])
        expected_cutoffs = np.array([20.] + [15.] * (n_channels - 2) + [5.])
        expected_thresholds = np.array([1.] + [0.8] * (n_channels - 2) + [0.4])
        state = fake.processing_state
        np.testing.assert_allclose(state.thresholds, expected_thresholds)
        np.testing.assert_allclose(state.T @ state.calibration_patterns,
                                   np.diag(expected_thresholds), atol=1e-15)
        np.testing.assert_allclose(details["calibration"]["component_cutoffs"], expected_cutoffs)
        np.testing.assert_allclose(details["calibration"]["eigenvalues"],
                                   [1.] + [4.] * (n_channels - 2) + [16.])
        np.testing.assert_allclose(details["calibration"]["thresholds"], expected_thresholds)
        np.testing.assert_allclose(plotting.call_args.args[3].thresholds, expected_thresholds)
        self.assertEqual(details["settings"]["cutoff"], 20.0)
        self.assertEqual(details["settings"]["cutoff_min"], 5.0)
        self.assertEqual(details["settings"]["cutoff_scaling"], "sqrt_eigenvalue")
        self.assertFalse(details["settings"]["normalized"])

    def test_normalized_uses_fixed_20_and_restores_electrode_coordinates(self):
        raw = make_raw()
        raw.info["bads"] = ["F7"]
        original = raw.get_data()
        selected = [i for i, name in enumerate(raw.ch_names)
                    if name in NAMES and name not in ["F7", "Cz"]]
        fake = FakeASR()
        calibrate = fake.calibrate

        def calibrate_with_spectrum(X, sfreq, **kwargs):
            state, diagnostics = calibrate(X, sfreq, **kwargs)
            n_channels = X.shape[0]
            eigenvalues = np.linspace(1., 16., n_channels)
            patterns = np.eye(n_channels)
            patterns[np.ix_([0, n_channels - 1], [0, n_channels - 1])] = [
                [0.6, -0.8], [0.8, 0.6],
            ]
            diagnostics["threshold_mu"] = np.linspace(0.2, 0.4, n_channels)
            return replace(
                state, cov=(patterns * eigenvalues) @ patterns.T,
                M=(patterns * np.sqrt(eigenvalues)) @ patterns.T,
                calibration_patterns=patterns,
            ), diagnostics

        fake.calibrate = calibrate_with_spectrum
        with patch.object(step2, "plot_asr_calibration") as plotting, \
                patch.object(step2, "plot_asr_effect"):
            report, details = self.run_fake(
                raw, fake, reference_channel="Cz", normalized=True, return_details=True,
            )
        state = plotting.call_args.args[3]
        thresholds = np.linspace(0.2, 0.4, len(selected)) + 20 * 0.04
        inverse = state.calibration_patterns * thresholds
        np.testing.assert_allclose(details["calibration"]["component_cutoffs"], 20.)
        np.testing.assert_allclose(details["calibration"]["thresholds"], thresholds)
        np.testing.assert_allclose(inverse @ fake.processing_data, original[selected])
        np.testing.assert_allclose(raw._data[selected], inverse @ fake.corrected)
        # The temporary processing state preserves the electrode-space prior.
        np.testing.assert_allclose(inverse @ fake.processing_state.M, state.M, atol=1e-14)
        np.testing.assert_allclose(fake.processing_state.T, np.eye(len(selected)))
        b, a = details["filter_b"], details["filter_a"]
        data = original[selected]
        history = 2 * data[:, [0]] - data[:, len(b) - 1:0:-1]
        _, zi = lfilter(b, a, history, axis=1,
                        zi=np.zeros((len(selected), len(b) - 1)))
        np.testing.assert_allclose(inverse @ fake.processing_state.filter_zi, zi,
                                   rtol=1e-10, atol=1e-15)
        for name in ["F7", "Cz", "EOG"]:
            row = raw.ch_names.index(name)
            np.testing.assert_array_equal(raw._data[row], original[row])
        self.assertEqual(report["component_segments"], [(100, 200), (300, 400)])
        self.assertEqual(report["corrected_segments"], [(100, 250), (300, 450)])
        self.assertEqual(details["settings"]["cutoff_min"], 20.)
        self.assertEqual(details["settings"]["cutoff_scaling"], "fixed")
        self.assertTrue(details["settings"]["normalized"])
        self.assertEqual(details["processing"]["component_units"], "dimensionless")

    def test_equal_eigenvalues_keep_cutoff_20(self):
        raw = make_raw()
        fake = FakeASR()
        with patch.object(step2, "plot_asr_calibration"), \
                patch.object(step2, "plot_asr_effect"):
            _, details = self.run_fake(raw, fake, reference_channel="Cz", return_details=True)
        np.testing.assert_array_equal(details["calibration"]["component_cutoffs"], 20.0)
        np.testing.assert_array_equal(fake.processing_state.thresholds, 1.0)
        np.testing.assert_array_equal(fake.processing_state.T, np.eye(len(details["ch_names"])))

    def test_post_asr_repair_uses_corrected_donors_and_protects_effective_reference(self):
        raw = make_raw(reference="F3")
        raw.info["bads"] = ["Cz", "F7"]
        flatlines = flatlines_for(raw, {"Fp1": [[1000, 1200]], "F3": [[1100, 1300]]})
        original = raw.get_data()
        fake = FakeASR()
        self.run_fake(raw, fake, reference_channel="F3", flatlines=flatlines)

        selected = [i for i, name in enumerate(raw.ch_names)
                    if name in NAMES and name not in ["Cz", "F7", "F3"]]
        corrected = original.copy()
        corrected[selected] = fake.corrected
        geometry = step2._spline_geometry(raw)
        target = raw.ch_names.index("Fp1")
        reference = raw.ch_names.index("F3")
        for start, stop, invalid in [
            (1000, 1100, [target]), (1100, 1200, [target, reference]),
        ]:
            donors = [i for i in geometry["picks"]
                      if raw.ch_names[i] not in raw.info["bads"] and i not in invalid]
            weights = step2._spline_weights(geometry, donors, [target])
            np.testing.assert_allclose(raw._data[target, start:stop],
                                       (weights @ corrected[donors, start:stop])[0],
                                       rtol=0, atol=1e-15)
        np.testing.assert_array_equal(raw._data[reference], 0)
        np.testing.assert_array_equal(raw._data[target, :1000], corrected[target, :1000])
        np.testing.assert_array_equal(raw._data[target, 1200:], corrected[target, 1200:])

    def test_real_asr_preserves_length_reference_and_excluded_channels(self):
        for normalized in [False, True]:
            with self.subTest(normalized=normalized):
                self.check_real_asr(normalized)

    def check_real_asr(self, normalized):
        raw = make_raw(seconds=90)
        # Independent bandpassed channels plus a large, brief artifact.
        rng = np.random.default_rng(914)
        data = sosfiltfilt(butter(4, [1, 30], btype="bandpass", fs=250, output="sos"),
                          rng.normal(size=(len(NAMES), raw.n_times)), axis=1)
        data *= 5e-6 / data.std(axis=1, keepdims=True)
        data -= data[NAMES.index("Cz")]
        raw._data[:len(NAMES)] = data
        raw.info["bads"] = ["F7"]
        artifact_rows = [raw.ch_names.index(name) for name in ["Fp1", "Fp2", "F3"]]
        start, stop = 55 * 250, 56 * 250
        raw._data[artifact_rows, start:stop] += (
            4e-3 * np.sin(2 * np.pi * 3 * np.arange(stop - start) / 250)
        )
        raw.set_annotations(mne.Annotations([5], [1], ["bad_acquisition"]))
        original = raw.get_data()
        designer = _calibration._design_statistics_filter
        report, details = step2.apply_asr_inplace(
            raw, "Cz", flatlines=flatlines_for(raw, {}),
            normalized=normalized, return_details=True,
        )

        self.assertEqual(raw.get_data().shape, original.shape)
        self.assertTrue(np.isfinite(raw.get_data()).all())
        np.testing.assert_array_equal(raw._data[raw.ch_names.index("Cz")], 0)
        for name in ["F7", "EOG"]:
            row = raw.ch_names.index(name)
            np.testing.assert_array_equal(raw._data[row], original[row])
        self.assertGreaterEqual(len(details["calibration_sample_indices"]), 60 * 250)
        self.assertEqual(details["calibration"]["blocksize"], 25)
        self.assertEqual(len(details["filter_b"]), 1001)
        self.assertEqual(len(details["filter_a"]), 1)
        self.assertFalse(np.isin(np.arange(5 * 250, 6 * 250),
                                details["calibration_input_sample_indices"]).any())
        self.assertTrue(report["corrected_segments"])
        self.assertLess(np.max(np.abs(raw._data[artifact_rows, start:stop])),
                        np.max(np.abs(original[artifact_rows, start:stop])))
        self.assertIs(_calibration._design_statistics_filter, designer)
        self.assertEqual(set(raw.annotations.description),
                         {"bad_acquisition", "BAD_voltage_mask"})


class TestNormalizedASR(unittest.TestCase):
    def test_rejects_mixtures_of_ordinary_empirical_components(self):
        # Both ordinary components pass their allowance of 10, but their
        # contrast has variance 2.5 against allowance 1.
        allowance = np.array([[10., 9.], [9., 10.]])
        values, patterns = np.linalg.eigh(allowance)
        thresholds = np.sqrt(values)
        M = (patterns * [0.5, 1.]) @ patterns.T
        state = ASRState(
            M=M, T=thresholds[:, None] * patterns.T,
            thresholds=thresholds, calibration_patterns=patterns,
            filter_b=np.ones(1), filter_a=np.ones(1), cov=M @ M.T, rank=2,
        )
        data = np.zeros((2, 2000))
        t = np.arange(1000) / 1000
        data[:, 500:1500] = np.sqrt(2) * np.array([
            2 * np.sin(2 * np.pi * 8 * t), np.sin(2 * np.pi * 16 * t),
        ])
        original_T = state.T.copy()
        standard, std_details = step2.process_asr(data, 1000, state)
        normalized, norm_details = step2._process_normalized_asr(data, 1000, state)
        middle = ((norm_details["window_starts"] >= 750)
                  & (norm_details["window_stops"] <= 1250))
        self.assertTrue(middle.any())
        np.testing.assert_array_equal(std_details["n_components_reconstructed"][middle], 0)
        np.testing.assert_array_equal(norm_details["n_components_reconstructed"][middle], 1)
        np.testing.assert_allclose(norm_details["component_variances"][middle, -1],
                                   2.54898679, rtol=1e-8)
        np.testing.assert_allclose(norm_details["component_thresholds"], 1., atol=1e-14)
        np.testing.assert_allclose(standard[:, 750:1250], data[:, 750:1250])
        self.assertGreater(np.linalg.norm(normalized[:, 750:1250] - data[:, 750:1250]), 1.)
        np.testing.assert_array_equal(state.T, original_T)

    def test_invalid_thresholds_raise_before_processing(self):
        state, _ = FakeASR().calibrate(np.ones((2, 1000)), 250, filter_kind="none",
                                       blocksize=25)
        for threshold in [0., -1., np.nan, np.inf]:
            with self.subTest(threshold=threshold), \
                    patch.object(step2, "process_asr") as processing:
                with self.assertRaisesRegex(ValueError, "finite, positive"):
                    step2._process_normalized_asr(
                        np.ones((2, 1000)), 250,
                        replace(state, thresholds=np.array([threshold, 1.])),
                    )
                processing.assert_not_called()


class TestASRFilter(unittest.TestCase):
    def test_fft_filter_output_and_state_match_scipy_lfilter(self):
        rng = np.random.default_rng(713)
        b, a = step2._asr_statistics_filter(250)
        order = len(b) - 1
        for n_times in [37, 1500]:
            with self.subTest(n_times=n_times):
                data = rng.normal(size=(3, n_times))
                zi = rng.normal(size=(3, order))
                expected, expected_state = lfilter(b, a, data, axis=1, zi=zi)
                actual, state = step2._asr_filter_channels(data, b, a, zi=zi)
                np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
                np.testing.assert_allclose(state, expected_state, rtol=1e-11, atol=1e-12)

                # Missing history is the reflected beginning of this sequence.
                history = 2 * data[:, [0]] - data[:, order:0:-1]
                combined = np.concatenate([history, data], axis=1)
                expected, expected_state = lfilter(
                    b, a, combined, axis=1, zi=np.zeros((3, order)),
                )
                actual, state = step2._asr_filter_channels(data, b, a)
                np.testing.assert_allclose(actual, expected[:, history.shape[1]:],
                                           rtol=1e-11, atol=1e-12)
                np.testing.assert_allclose(state, expected_state, rtol=1e-11, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
