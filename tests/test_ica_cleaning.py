import pickle
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import mne
import numpy as np
from mne_icalabel.iclabel.features import _eeg_autocorr_fftw, _eeg_rpsd


pipeline_dir = Path(__file__).resolve().parents[1]
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

import step3_ica_cleaning as step3


NAMES = ["E14", "E126", "E21", "E127", "F3", "C3", "Cz", "BAD", "EOG"]


def make_raw(seconds=12, sfreq=1000, first_samp=5500):
    data = np.random.default_rng(518).normal(scale=10e-6,
                                            size=(len(NAMES), int(seconds * sfreq)))
    data[NAMES.index("Cz")] = 0
    data[NAMES.index("BAD")] = np.nan
    raw = mne.io.RawArray(
        data, mne.create_info(NAMES, sfreq, ["eeg"] * 8 + ["eog"]),
        first_samp=first_samp, verbose=False,
    )
    raw.info["bads"] = ["BAD"]
    positions = {name: position for name, position in zip(NAMES[:8], [
        [-0.04, 0.07, 0.04], [0.04, 0.07, 0.04], [-0.06, 0.02, 0.06],
        [0.06, 0.02, 0.06], [-0.04, 0.03, 0.075], [-0.045, 0, 0.075],
        [0, 0, 0.09], [0.04, -0.03, 0.075],
    ])}
    raw.set_montage(mne.channels.make_dig_montage(positions, coord_frame="head"))
    return raw


def prepare(raw, reference_channel=None):
    with warnings.catch_warnings(), mne.use_log_level("ERROR"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return step3.prepare_ica_epochs(raw, reference_channel)


class FakeICA:
    def __init__(self, n_components):
        self.n_components_ = n_components
        self.n_iter_ = 9
        self.ch_names = ["E14", "E126", "F3"]
        self.exclude = [99]
        self.maps = np.eye(3, n_components)
        self.fit = Mock(return_value=self)

    def get_components(self):
        return self.maps


class TestICAPreparation(unittest.TestCase):
    def test_masks_and_decimation_preserve_original_native_data(self):
        raw = make_raw()
        raw.set_annotations(mne.Annotations(
            [2.1, 4.1, 8.1], [0.2, 0.2, 0.2],
            ["BAD_voltage_mask", "BAD_by_flatline", "local_repaired"],
        ))
        original = raw.copy()
        filter_calls = []
        original_filter = mne.io.BaseRaw.filter

        def record_filter(instance, *args, **kwargs):
            filter_calls.append(kwargs.copy())
            return original_filter(instance, *args, **kwargs)

        with patch.object(mne.io.BaseRaw, "filter", record_filter):
            raw_ica, epochs, reference = prepare(raw)

        self.assertEqual(reference, "Cz")
        self.assertIsNot(raw_ica, raw)
        self.assertEqual(raw_ica.ch_names, NAMES[:7])
        self.assertEqual(epochs.ch_names, NAMES[:6])
        self.assertEqual(epochs.get_data().shape, (4, 6, 500))
        self.assertEqual(epochs.info["sfreq"], 250)
        self.assertIsNone(epochs.baseline)
        self.assertFalse(epochs.proj)
        np.testing.assert_array_equal(epochs.events[:, 0], raw.first_samp +
                                      np.array([0, 6, 8, 10]) * 1000)
        np.testing.assert_array_equal(raw_ica.get_data(picks=["Cz"]), 0)
        self.assertTrue(np.isfinite(epochs.get_data()).all())
        self.assertEqual([(call["l_freq"], call["h_freq"]) for call in filter_calls],
                         [(1.0, None), (None, 100.0)])
        self.assertEqual(filter_calls[1]["h_trans_bandwidth"], 25)
        for call in filter_calls:
            self.assertEqual(call["skip_by_annotation"], ("edge", "bad_acq_skip", "BAD"))
        np.testing.assert_array_equal(raw.get_data(), original.get_data())
        self.assertEqual(raw.ch_names, original.ch_names)
        self.assertEqual(raw.info["bads"], original.info["bads"])
        self.assertEqual(raw.annotations, original.annotations)
        self.assertEqual(raw.first_samp, original.first_samp)
        for key in ("highpass", "lowpass", "sfreq", "custom_ref_applied", "projs"):
            self.assertEqual(raw.info[key], original.info[key])

    def test_shifted_effective_reference_is_selected(self):
        raw = make_raw()
        eeg = mne.pick_types(raw.info, eeg=True, exclude=[])
        raw._data[eeg] -= raw._data[NAMES.index("C3")].copy()
        raw.info["bads"].append("Cz")
        for supplied_reference in (None, "C3"):
            with self.subTest(reference=supplied_reference):
                raw_ica, epochs, reference = prepare(raw, supplied_reference)
                self.assertEqual(reference, "C3")
                self.assertIn("C3", raw_ica.ch_names)
                self.assertNotIn("C3", epochs.ch_names)
                self.assertNotIn("Cz", raw_ica.ch_names)

    def test_requires_the_native_rate_and_zero_reference(self):
        with self.assertRaisesRegex(ValueError, "1000"):
            prepare(make_raw(sfreq=500))
        with self.assertRaisesRegex(ValueError, "zero"):
            prepare(make_raw(), "F3")
        raw = make_raw()
        raw._data[NAMES.index("C3")] = 0
        with self.assertRaisesRegex(ValueError, "reference"):
            prepare(raw)

    def test_rejects_training_when_no_clean_two_second_epoch_remains(self):
        raw = make_raw()
        raw.set_annotations(mne.Annotations(np.arange(0, 12, 2) + 0.5,
                                            [0.1] * 6, ["BAD_voltage_mask"] * 6))
        with self.assertRaisesRegex(ValueError, "No clean 2-second epochs"):
            prepare(raw)


class TestICLabelMaps(unittest.TestCase):
    def test_restores_reference_and_scales_activations_before_time_features(self):
        raw = make_raw().pick(["E14", "E126", "F3", "Cz"])
        epochs = mne.make_fixed_length_epochs(raw.copy().drop_channels(["Cz"]),
                                             duration=2, preload=True, verbose=False)
        ica = FakeICA(3)
        ica.maps = np.array([[2, 0.3, -0.7], [0.1, 0.8, -0.2], [0, 0, 3]])
        original_maps = ica.maps.copy()
        original_data = epochs.get_data().copy()
        activations = np.arange(3 * len(epochs.times) * len(epochs), dtype=float).reshape(
            3, len(epochs.times), len(epochs),
        )
        original_activations = activations.copy()
        psd = np.full((1, 100, 1, 3), 0.1, dtype=np.float32)
        autocorr = np.full((1, 100, 1, 3), 0.2, dtype=np.float32)
        psd_original, autocorr_original = psd.copy(), autocorr.copy()
        new_topo = np.full((32, 32, 1, 3), 0.5, dtype=np.float32)
        probabilities = np.zeros((3, 7))
        probabilities[:, 0] = 1
        with patch.object(step3, "_compute_ica_activations",
                          side_effect=lambda *_: activations.copy()) as compute, \
                patch.object(step3, "_eeg_rpsd", return_value=psd) as rpsd, \
                patch.object(step3, "_eeg_autocorr_fftw",
                             return_value=autocorr) as ac, \
                patch.object(step3, "_eeg_topoplot", return_value=new_topo) as topoplot, \
                patch.object(step3, "run_iclabel", return_value=probabilities) as infer:
            result = step3.label_ica_components(raw, epochs, ica, "Cz")

        self.assertIs(result, probabilities)
        compute.assert_called_once_with(epochs, ica)
        map_raw, maps, names = topoplot.call_args.args
        self.assertEqual(names, ica.ch_names + ["Cz"])
        expected = np.vstack([original_maps, np.zeros((1, 3))])
        expected -= expected.mean(axis=0, keepdims=True)
        np.testing.assert_array_equal(maps, expected)
        np.testing.assert_allclose(maps.sum(axis=0), 0, atol=1e-15)
        self.assertEqual(np.linalg.matrix_rank(maps), 3)
        for name in names:
            np.testing.assert_array_equal(
                map_raw.info["chs"][map_raw.ch_names.index(name)]["loc"],
                raw.info["chs"][raw.ch_names.index(name)]["loc"],
            )
        np.testing.assert_allclose(infer.call_args.args[0], new_topo * 0.99)
        for extractor in (rpsd, ac):
            extractor.assert_called_once()
            self.assertIs(extractor.call_args.args[0], epochs)
            self.assertIs(extractor.call_args.args[1], ica)
            np.testing.assert_allclose(
                extractor.call_args.args[2],
                original_activations * np.sqrt(np.mean(expected ** 2, axis=0))[:, None, None],
            )
        np.testing.assert_array_equal(infer.call_args.args[1], psd * 0.99)
        np.testing.assert_array_equal(infer.call_args.args[2], autocorr * 0.99)
        self.assertEqual(infer.call_args.kwargs, {"backend": None})
        np.testing.assert_array_equal(psd, psd_original)
        np.testing.assert_array_equal(autocorr, autocorr_original)
        np.testing.assert_array_equal(ica.maps, original_maps)
        np.testing.assert_array_equal(epochs.get_data(), original_data)
        self.assertEqual(ica.exclude, [99])


class TestICLabelFeatureScaling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = make_raw(seconds=8, sfreq=250).pick(NAMES[:7])
        cls.epochs = mne.make_fixed_length_epochs(
            cls.raw.copy().drop_channels(["Cz"]), duration=2, preload=True, verbose=False,
        )
        cls.ica = mne.preprocessing.ICA(
            n_components=6, method="infomax", fit_params={"extended": True},
            max_iter=32, rng=518,
        )
        with warnings.catch_warnings(), mne.use_log_level("ERROR"):
            warnings.simplefilter("ignore", RuntimeWarning)
            cls.ica.fit(cls.epochs, picks="eeg")
        cls.ica.exclude = [1]

    def features(self, ica):
        with warnings.catch_warnings(), patch.object(
            step3, "run_iclabel", side_effect=lambda *args, **_: args,
        ):
            warnings.simplefilter("ignore", RuntimeWarning)
            return step3.label_ica_components(self.raw, self.epochs, ica, "Cz")

    def test_time_features_match_independent_physical_car_inversion(self):
        # Normalize physical CAR mixing columns as eeg_checkset does, then
        # invert those maps against CAR data in microvolts independently.
        physical_maps = np.vstack([
            self.ica.get_components() * self.ica.pre_whitener_ * 1e6,
            np.zeros((1, self.ica.n_components_)),
        ])
        physical_maps -= physical_maps.mean(axis=0, keepdims=True)
        physical_maps /= np.sqrt(np.mean(physical_maps ** 2, axis=0))
        data = np.concatenate([
            self.epochs.get_data() * 1e6,
            np.zeros((len(self.epochs), 1, len(self.epochs.times))),
        ], axis=1)
        data -= data.mean(axis=1, keepdims=True)
        activations = (np.linalg.pinv(physical_maps) @ data).transpose(1, 2, 0)
        _, psd, autocorr = self.features(self.ica)
        np.testing.assert_allclose(
            psd, _eeg_rpsd(self.epochs, self.ica, activations) * 0.99, atol=2e-6,
        )
        np.testing.assert_allclose(
            autocorr, _eeg_autocorr_fftw(self.epochs, self.ica, activations) * 0.99,
            atol=2e-6,
        )

    def test_features_are_invariant_to_inverse_map_and_source_rescaling(self):
        baseline = self.features(self.ica)
        sources = self.ica.get_sources(self.epochs).get_data()
        for gains in (np.geomspace(0.01, 100, 6), np.array([1, -10, 1, 1, 1, 1])):
            with self.subTest(gains=gains):
                rescaled = self.ica.copy()
                rescaled.mixing_matrix_ *= gains[None, :]
                rescaled.unmixing_matrix_ /= gains[:, None]
                np.testing.assert_allclose(
                    rescaled.get_components(), self.ica.get_components() * gains[None, :],
                )
                np.testing.assert_allclose(
                    rescaled.get_sources(self.epochs).get_data() * gains[None, :, None],
                    sources, atol=1e-12,
                )
                topo, psd, autocorr = self.features(rescaled)
                np.testing.assert_allclose(
                    topo * np.sign(gains)[None, None, None, :], baseline[0], atol=2e-6,
                )
                np.testing.assert_allclose(psd, baseline[1], atol=2e-6)
                np.testing.assert_allclose(autocorr, baseline[2], atol=2e-6)

    def test_labeling_preserves_fitted_ica_and_inputs(self):
        ica_state = pickle.dumps(self.ica)
        raw_data, epochs_data = self.raw.get_data().copy(), self.epochs.get_data().copy()
        raw_info, epochs_info = pickle.dumps(self.raw.info), pickle.dumps(self.epochs.info)
        self.features(self.ica)
        self.assertEqual(pickle.dumps(self.ica), ica_state)
        np.testing.assert_array_equal(self.raw.get_data(), raw_data)
        np.testing.assert_array_equal(self.epochs.get_data(), epochs_data)
        self.assertEqual(pickle.dumps(self.raw.info), raw_info)
        self.assertEqual(pickle.dumps(self.epochs.info), epochs_info)


class TestICASelection(unittest.TestCase):
    def test_exclusion_helper_returns_sorted_unique_indices_and_unrounded_table(self):
        probabilities = np.zeros((4, 7))
        probabilities[0, [0, 1]] = [0.85, 0.15]
        probabilities[1, [0, 1]] = [0.24999, 0.75001]
        probabilities[2, [0, 6]] = [0.6, 0.4]
        probabilities[3, [0, 2]] = [0.2, 0.8]
        original = probabilities.copy()
        with patch.object(step3, "display") as display:
            exclusions, table = step3.exclude_ica_components(probabilities)

        self.assertEqual(exclusions, [1, 3])
        np.testing.assert_array_equal(probabilities, original)
        class_names = ["brain", "muscle artifact", "eye blink", "heart beat", "line noise",
                       "channel noise", "other"]
        self.assertEqual(table.columns.tolist(),
                         ["component", "winning_label", "winning_probability", *class_names,
                          "iclabel_auto_exclude", "excluded"])
        np.testing.assert_array_equal(table[class_names].to_numpy(), original)
        self.assertEqual(table["component"].tolist(), [0, 1, 2, 3])
        self.assertEqual(table["excluded"].tolist(), [False, True, False, True])
        self.assertEqual(table.loc[1, "winning_probability"], 0.75001)
        display.assert_called_once()
        self.assertEqual(display.call_args.args[0].loc[1, "winning_probability"], 0.75)

    def fit_with_probabilities(self, probabilities):
        raw = make_raw()
        raw_ica = raw.copy().pick("eeg", exclude="bads")
        epochs = mne.make_fixed_length_epochs(raw_ica, duration=2, preload=True, verbose=False)
        ica = FakeICA(len(probabilities))
        patches = (
            patch.object(step3, "prepare_ica_epochs", return_value=(raw_ica, epochs, "Cz")),
            patch.object(step3.mne.preprocessing, "ICA", return_value=ica),
            patch.object(step3, "label_ica_components", return_value=probabilities),
            patch.object(step3, "display"),
        )
        return raw, epochs, ica, patches

    def test_strict_threshold_and_full_precision_table(self):
        probabilities = np.zeros((7, 7))
        probabilities[0, [0, 1]] = [0.9, 0.1]
        probabilities[1, [0, 1]] = [0.24999, 0.75001]
        probabilities[2, [0, 2]] = [0.25, 0.75]
        probabilities[3, [0, 6]] = [0.25, 0.75]
        probabilities[4, [0, 4]] = [0.2, 0.8]
        probabilities[5, [0, 6]] = [0.1, 0.9]
        probabilities[6, [0, 1, 2]] = [0.2, 0.4, 0.4]
        raw, epochs, ica, patches = self.fit_with_probabilities(probabilities)
        original = raw.copy()
        with patches[0] as prep, patches[1] as constructor, patches[2] as label, patches[3] as display:
            returned_ica, fit_epochs, table = step3.fit_ica(raw, "Cz", return_details=True)
        self.assertIs(returned_ica, ica)
        self.assertIs(fit_epochs, epochs)
        self.assertEqual(ica.exclude, [1, 4])
        prep.assert_called_once_with(raw, "Cz")
        constructor.assert_called_once_with(
            n_components=None, method="infomax", fit_params={"extended": True},
            max_iter="auto", rng=518,
        )
        ica.fit.assert_called_once_with(epochs, picks="eeg")
        label.assert_called_once_with(prep.return_value[0], epochs, ica, "Cz")
        self.assertEqual(prep.return_value[0].ch_names, NAMES[:7])
        class_names = ["brain", "muscle artifact", "eye blink", "heart beat", "line noise",
                       "channel noise", "other"]
        np.testing.assert_array_equal(table[class_names].to_numpy(), probabilities)
        self.assertEqual(table["winning_label"].tolist(),
                         ["brain", "muscle artifact", "eye blink", "other", "line noise", "other",
                          "muscle artifact"])
        self.assertEqual(table["iclabel_auto_exclude"].tolist(),
                         [False, True, False, False, True, False, False])
        self.assertEqual(table["excluded"].tolist(),
                         [False, True, False, False, True, False, False])
        self.assertEqual(display.call_args.args[0].loc[1, "muscle artifact"], 0.75)
        self.assertEqual(table.loc[1, "muscle artifact"], 0.75001)
        np.testing.assert_array_equal(raw.get_data(), original.get_data())
        self.assertEqual(raw.info["bads"], original.info["bads"])

    def test_brain_and_other_probabilities_do_not_trigger_exclusion(self):
        for column, category in ((0, "brain"), (6, "other")):
            with self.subTest(category=category):
                probabilities = np.zeros((1, 7))
                probabilities[0, column] = 0.9
                probabilities[0, 2] = 0.1
                raw, epochs, ica, patches = self.fit_with_probabilities(probabilities)
                with patches[0], patches[1], patches[2], patches[3] as display:
                    returned_ica, fit_epochs = step3.fit_ica(raw)
                self.assertIs(returned_ica, ica)
                self.assertIs(fit_epochs, epochs)
                display.assert_called_once()
                self.assertEqual(ica.exclude, [])

    def test_custom_threshold_reaches_exclusion_decisions(self):
        probabilities = np.zeros((3, 7))
        probabilities[0, [0, 1]] = [0.35, 0.65]
        probabilities[1, [0, 2]] = [0.4, 0.6]
        probabilities[2, [0, 6]] = [0.6, 0.4]
        raw, epochs, ica, patches = self.fit_with_probabilities(probabilities)
        with patches[0], patches[1], patches[2], patches[3]:
            returned_ica, fit_epochs, table = step3.fit_ica(
                raw, iclabel_threshold=0.6, return_details=True,
            )
        self.assertIs(returned_ica, ica)
        self.assertIs(fit_epochs, epochs)
        self.assertEqual(ica.exclude, [0])
        self.assertEqual(table["iclabel_auto_exclude"].tolist(), [True, False, False])


if __name__ == "__main__":
    unittest.main()
