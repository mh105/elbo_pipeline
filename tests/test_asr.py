import sys
import unittest
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
    """Supply prescribed gap-joined channel intervals."""
    return {
        "ch_names": list(spans),
        "segments_exact": [np.asarray(value, dtype=int).reshape(-1, 2)
                           for value in spans.values()],
        "segments_bridged": [np.asarray(value, dtype=int).reshape(-1, 2)
                     for value in spans.values()],
        "raw_ch_names": list(raw.ch_names), "eeg_ch_names": NAMES.copy(),
        "sfreq": float(raw.info["sfreq"]), "first_samp": int(raw.first_samp),
        "n_times": int(raw.n_times), "reference_channel": reference,
        "global_segments_exact": [],
    }


class FakeASR:
    """Control ASR decisions while retaining its actual state/data interfaces."""

    def __init__(self, retained=None):
        self.retained = retained
        self.calibration_data = None
        self.processing_data = None
        self.calibration_kwargs = None
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
        _, filter_zi = _calibration._lfilter_channels(X[:, retained], b, a)
        state = ASRState(
            M=np.eye(n_channels), T=np.eye(n_channels),
            thresholds=np.ones(n_channels), calibration_patterns=np.eye(n_channels),
            filter_b=b, filter_a=a, cov=np.eye(n_channels), rank=n_channels,
            filter_zi=filter_zi,
        )
        return state, {"clean_sample_mask": retained,
                       "calibration_samples": int(retained.sum()),
                       "blocksize": kwargs["blocksize"]}

    def process(self, X, sfreq, state, **kwargs):
        self.processing_data = X.copy()
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
                patch.object(step2, "process_asr", side_effect=fake.process), \
                patch.object(step2, "bad_ch_by_pop", side_effect=AssertionError(
                    "ASR must not redetect pops."
                )), \
                patch.object(step2, "interpolate_local_artifacts_inplace", side_effect=AssertionError(
                    "ASR output must not be interpolated again."
                )), \
                patch.object(step2, "mask_segment_by_voltage", side_effect=AssertionError(
                    "Prepare voltage mask annotations explicitly before ASR."
                )):
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
        details = self.run_fake(
            raw, fake, reference_channel="F3", flatlines=flatlines,
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
        self.assertEqual(fake.calibration_kwargs["cutoff"], 5)
        self.assertEqual(details["settings"]["cutoff"], 5.0)
        self.assertEqual(fake.calibration_kwargs["blocksize"], 25)
        self.assertEqual(details["ch_names"], [raw.ch_names[i] for i in selected])
        self.assertEqual(details["reference_channel"], "F3")
        # Global exclusions affect calibration only: correction reaches them too.
        np.testing.assert_array_equal(raw._data[selected, 250:500],
                                      fake.corrected[:, 250:500])
        reference = raw.ch_names.index("F3")
        np.testing.assert_array_equal(raw._data[selected], fake.corrected)
        for name in ["F7", "Cz"]:
            row = raw.ch_names.index(name)
            np.testing.assert_array_equal(raw._data[row], original[row])
        np.testing.assert_array_equal(raw._data[reference], 0)
        np.testing.assert_array_equal(raw._data[-1], original[-1])
        self.assertEqual(raw.annotations, annotations)
        self.assertEqual(raw.info["bads"], ["F7", "Cz"])
        self.assertEqual(details["component_segments"], [(100, 200), (300, 400)])
        self.assertEqual(details["corrected_segments"], [(100, 250), (300, 450)])
        self.assertEqual(details["duration_s"], 75.)
        self.assertEqual(details["corrected_duration_s"], 1.2)
        self.assertEqual(details["corrected_percent"], 1.6)
        for segments in [details["corrected_segments"], details["component_segments"]]:
            self.assertTrue(all(type(start) is int and type(stop) is int
                                for start, stop in segments))

    def test_custom_calibration_cutoff_is_used_and_reported(self):
        raw = make_raw()
        fake = FakeASR()
        details = self.run_fake(
            raw, fake, reference_channel="Cz", calibration_cutoff=7.5,
        )
        self.assertEqual(fake.calibration_kwargs["cutoff"], 7.5)
        self.assertEqual(details["settings"]["cutoff"], 7.5)

    def test_pop_calibration_exclusions_use_exact_local_and_reference_intervals(self):
        raw = make_raw(sfreq=100, reference="F3")
        raw.info["bads"] = ["F7", "Cz"]
        raw.set_annotations(mne.Annotations([1], [1], ["BAD_manual"]))
        flatlines = flatlines_for(raw, {
            "Fp1": [[210, 230]], "F3": [[270, 280]],
        })
        pop_intervals = {
            "Fp2": [(150, 220), (232, 240), (500, 520), (522, 530)],
            "F3": [(275, 290), (600, 610)],
            "F7": [(0, raw.n_times)], "Cz": [(0, raw.n_times)],
        }
        original = raw.get_data()
        excluded = np.zeros(raw.n_times, dtype=bool)
        for start, stop in [(100, 230), (232, 240), (270, 290),
                            (500, 520), (522, 530), (600, 610)]:
            excluded[start:stop] = True
        expected_indices = np.flatnonzero(~excluded)
        selected = [i for i, name in enumerate(raw.ch_names)
                    if name in NAMES and name not in ["F7", "Cz", "F3"]]
        fake = FakeASR()

        details = self.run_fake(
            raw, fake, reference_channel="F3", flatlines=flatlines,
            pop_intervals=pop_intervals,
        )

        np.testing.assert_array_equal(details["calibration_input_sample_indices"],
                                      expected_indices)
        np.testing.assert_array_equal(fake.calibration_data,
                                      original[np.ix_(selected, expected_indices)])
        # Neither edges nor small gaps between local exclusions are expanded.
        for sample in [99, 230, 231, 240, 269, 290, 499, 520, 521, 530, 599, 610]:
            self.assertIn(sample, expected_indices)
        np.testing.assert_array_equal(fake.processing_data, original[selected])
        # Non-target channels still receive full-timeline ASR correction.
        unaffected = [i for i in selected if raw.ch_names[i] != "Fp2"]
        rows = [selected.index(i) for i in unaffected]
        np.testing.assert_array_equal(raw._data[unaffected, 500:530],
                                      fake.corrected[rows, 500:530])

    def test_asr_uses_explicit_voltage_mask_and_preserves_annotations(self):
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
        step2.mask_segment_by_voltage(raw, reference_channel="Cz")
        annotations = raw.annotations.copy()
        fake = FakeASR()
        details = self.run_fake(raw, fake, reference_channel="Cz")
        self.assertEqual(raw.annotations, annotations)
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

    def test_returns_details_and_blocksize_tracks_sampling_rate(self):
        for sfreq in [100, 250, 1000]:
            with self.subTest(sfreq=sfreq):
                raw = make_raw(sfreq=sfreq)
                fake = FakeASR()
                result = self.run_fake(raw, fake, reference_channel="Cz")
                self.assertIsInstance(result, dict)
                self.assertEqual(set(result), {
                    "duration_s", "corrected_duration_s", "corrected_percent",
                    "corrected_segments", "component_segments", "sfreq", "first_samp",
                    "ch_names", "reference_channel", "calibration_input_sample_indices",
                    "calibration_sample_indices", "filter_b", "filter_a", "settings",
                    "calibration", "processing",
                })
                self.assertEqual(result["duration_s"], 75.)
                self.assertEqual(result["corrected_duration_s"], 300 / sfreq)
                self.assertEqual(result["corrected_percent"], 100 * 300 / raw.n_times)
                for name in ["duration_s", "corrected_duration_s", "corrected_percent"]:
                    self.assertIs(type(result[name]), float)
                self.assertEqual(fake.calibration_kwargs["blocksize"], int(0.1 * sfreq))

    def test_no_correction_reports_zero_duration_and_percentage(self):
        raw = make_raw()
        fake = FakeASR()
        process = fake.process

        def identity_process(X, sfreq, state, **kwargs):
            _, diagnostics = process(X, sfreq, state, **kwargs)
            diagnostics["n_components_reconstructed"][:] = 0
            diagnostics["sample_mask"][:] = False
            return X.copy(), diagnostics

        fake.process = identity_process
        details = self.run_fake(raw, fake, reference_channel="Cz")
        self.assertEqual(details["duration_s"], 75.)
        self.assertEqual(details["corrected_duration_s"], 0.)
        self.assertEqual(details["corrected_percent"], 0.)
        self.assertEqual(details["corrected_segments"], [])
        self.assertEqual(details["component_segments"], [])

    def test_asr_output_is_preserved_with_local_flat_and_pop_intervals(self):
        raw = make_raw(reference="F3")
        raw.info["bads"] = ["Cz", "F7"]
        raw.set_annotations(mne.Annotations([5], [0.4], ["BAD_by_flatline"]))
        flatlines = flatlines_for(raw, {
            "Fp1": [[1000, 1400]], "F3": [[1000, 1300]],
            "F7": [[0, raw.n_times]],
        })
        pop_intervals = {
            "Fp1": [[1300, 1600]], "Fp2": [[1000, 1600]],
            "F3": [[1250, 1600]], "Cz": [[0, raw.n_times]],
        }
        original = raw.get_data()
        fake = FakeASR()
        details = self.run_fake(
            raw, fake, reference_channel="F3", flatlines=flatlines,
            pop_intervals=pop_intervals,
        )

        selected = [i for i, name in enumerate(raw.ch_names)
                    if name in NAMES and name not in ["Cz", "F7", "F3"]]
        expected = original.copy()
        expected[selected] = fake.corrected
        reference = raw.ch_names.index("F3")
        np.testing.assert_array_equal(raw.get_data(), expected)
        np.testing.assert_array_equal(raw._data[reference], 0)
        for name in ["Cz", "F7"]:
            row = raw.ch_names.index(name)
            np.testing.assert_array_equal(raw._data[row], original[row])
        np.testing.assert_array_equal(raw._data[-1], original[-1])
        self.assertEqual(details["corrected_segments"], [(100, 250), (300, 450)])
        self.assertEqual(details["component_segments"], [(100, 200), (300, 400)])
        self.assertEqual(details["reference_channel"], "F3")

    def test_reference_pop_intervals_only_exclude_calibration_samples(self):
        raw = make_raw()
        raw.info["bads"] = ["F7"]
        original = raw.get_data()
        fake = FakeASR()
        details = self.run_fake(
            raw, fake, reference_channel="Cz", pop_intervals={"Cz": [[1000, 1400]]},
        )

        selected = [i for i, name in enumerate(raw.ch_names)
                    if name in NAMES and name not in ["F7", "Cz"]]
        reference = raw.ch_names.index("Cz")
        expected = original.copy()
        expected[selected] = fake.corrected
        np.testing.assert_array_equal(raw.get_data(), expected)
        self.assertFalse(np.isin(np.arange(1000, 1400),
                                details["calibration_input_sample_indices"]).any())
        np.testing.assert_array_equal(raw._data[reference], 0)
        np.testing.assert_array_equal(raw._data[-1], original[-1])

    def test_identity_asr_preserves_pre_asr_joint_repair_and_outer_fades(self):
        raw = make_raw()
        raw.info["bads"] = ["F7"]
        flatlines = flatlines_for(raw, {"Fp1": [[1000, 1400]]})
        pop_intervals = {"Fp1": [[1300, 1600]]}
        original = raw.get_data()
        geometry = step2._spline_geometry(raw)
        target = raw.ch_names.index("Fp1")
        donors = [i for i in geometry["picks"]
                  if raw.ch_names[i] not in raw.info["bads"] and i != target]
        weights = step2._spline_weights(geometry, donors, [target])
        estimate = (weights @ original[donors, 1000:1600])[0]
        rise = 0.5 - 0.5 * np.cos(np.pi * (np.arange(62) + 0.5) / 62)
        blend = np.ones(600)
        blend[:62], blend[-62:] = rise, rise[::-1]
        source = original[target, 1000:1600]

        step2.interpolate_local_artifacts_inplace(
            raw, flatlines, pop_intervals=pop_intervals,
        )
        np.testing.assert_allclose(raw._data[target, 1000:1600],
                                   (1 - blend) * source + blend * estimate,
                                   rtol=0, atol=1e-15)
        repaired = raw.get_data()
        fake = FakeASR()
        process = fake.process

        def identity_process(X, sfreq, state, **kwargs):
            _, diagnostics = process(X, sfreq, state, **kwargs)
            diagnostics["n_components_reconstructed"][:] = 0
            diagnostics["sample_mask"][:] = False
            return X.copy(), diagnostics

        fake.process = identity_process
        details = self.run_fake(
            raw, fake, reference_channel="Cz", flatlines=flatlines,
            pop_intervals=pop_intervals,
        )

        np.testing.assert_array_equal(raw.get_data(), repaired)
        # The flat/pop overlap introduces no fades at either internal boundary.
        np.testing.assert_allclose(raw._data[target, 1299:1401], estimate[299:401],
                                   rtol=0, atol=1e-15)
        self.assertEqual(details["corrected_duration_s"], 0.)
        self.assertEqual(details["corrected_segments"], [])

    def test_real_asr_preserves_length_reference_and_excluded_channels(self):
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
        step2.mask_segment_by_voltage(raw, reference_channel="Cz")
        original = raw.get_data()
        designer = _calibration._design_statistics_filter
        details = step2.apply_asr_inplace(raw, "Cz", flatlines=flatlines_for(raw, {}))

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
        self.assertTrue(details["corrected_segments"])
        self.assertLess(np.max(np.abs(raw._data[artifact_rows, start:stop])),
                        np.max(np.abs(original[artifact_rows, start:stop])))
        self.assertIs(_calibration._design_statistics_filter, designer)
        self.assertEqual(set(raw.annotations.description),
                         {"bad_acquisition", "BAD_voltage_mask"})


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
